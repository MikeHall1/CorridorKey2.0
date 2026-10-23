"""Archived path admission is lexical on this host; no foreign filesystem claim."""
import json
import os
import unittest
from dataclasses import asdict
from pathlib import Path, PureWindowsPath
from typing import Any
from unittest.mock import patch

from corridorkey.config import InferenceConfig, WriterConfig
from corridorkey.diagnostics import json_value
from corridorkey.portable_paths import (
    MAX_PATH_BYTES,
    MAX_WIRE_BYTES,
    ArchivalJob,
    archive_job,
    archived_path,
    infer_flavour,
    native_destination,
)
from corridorkey.worker_protocol import job_from_wire


def archival_wire(flavour='windows') -> dict[str,Any]:
    root = 'C:\\shots' if flavour=='windows' else '/shots'
    separator = '\\' if flavour=='windows' else '/'
    settings = json_value(InferenceConfig(corrections_path=root+separator+'corrections.json',start_frame=5,num_frames=12))
    settings = json.loads(json.dumps(settings))
    settings['temp_dir'] = root+separator+'scratch'
    return {'input_path':root+separator+'scene'+separator+'frames','output_dir':root+separator+'output',
            'config_json':json.dumps(settings,indent=2),'writer_json':json.dumps(asdict(WriterConfig())),
            'initial_hint_path':root+separator+'first.png','hint_video_path':None,'profile_stages':False}


class PortablePathTests(unittest.TestCase):
    def test_windows_posix_unc_lexical_coordinates(self):
        self.assertEqual(infer_flavour(r'C:\shots\frames'),'windows')
        self.assertEqual(infer_flavour('/shots/frames'),'posix')
        self.assertEqual(archived_path(r'C:\shots\scene\frames','windows').parent.name,'scene')
        self.assertEqual(archived_path('D:/shots/scene/frames','windows').name,'frames')
        self.assertEqual(archived_path(r'\\server\share\scene\frames','windows').parts[0],'\\\\server\\share\\')
        self.assertIsInstance(archived_path(r'C:\shots\frame.exr','windows'),PureWindowsPath)
        self.assertEqual(archived_path('/shots/frame.exr','posix').stem,'frame')

    def test_rejects_ambiguous_relative_device_dot_uri_and_windows_unsafe_names(self):
        bad = ['relative','C:relative',r'\rootrelative','//server/share','https://server/shot',
               r'\\?\C:\shots',r'\\.\C:\shots','/shots/../other','/shots/./frame','/shots//frame',
               r'C:\shots/frames',r'C:\shots\..\other',r'C:\shots\NUL.exr',
               r'C:\shots\file:stream','C:\\shots\\trailing.','C:\\shots\\trailing ',
               r'\\server',r'\\server\\share','/shots/\x00frame']
        for value in bad:
            with self.subTest(value=value),self.assertRaises(ValueError):
                archived_path(value,infer_flavour(value))
        with self.assertRaises(ValueError):
            archived_path('/posix','windows')

    def test_archival_wire_and_embedded_bytes_preserved_without_native_path_io(self):
        wire = archival_wire()
        raw = json.dumps(wire,indent=3).encode()
        with (patch.object(Path,'resolve',side_effect=AssertionError('no native resolve')),
              patch.object(Path,'stat',side_effect=AssertionError('no native stat')),
              patch.object(Path,'open',side_effect=AssertionError('no native open'))):
            result = archive_job(raw)
            self.assertEqual(result.wire_json,raw)
            self.assertEqual(result.original_wire,wire)
            self.assertEqual(result.flavour,'windows')
            self.assertEqual(result.config.start_frame,5)
            self.assertEqual(result.config.temp_dir,r'C:\shots\scratch')
            self.assertEqual(result.config.corrections_path,r'C:\shots\corrections.json')
            self.assertEqual(result.writer.crf,12)
        changed = result.original_wire
        changed['input_path'] = '/changed'
        self.assertEqual(result.original_wire,wire)
        config = result.config
        config.outputs.clear()
        self.assertTrue(result.config.outputs)

    def test_native_wire_retains_shared_validation_and_default_relative_executable(self):
        wire = archival_wire('posix')
        archive = archive_job(wire)
        native = job_from_wire(wire)
        config,writer = native.configurations()
        self.assertEqual(archive.config.start_frame,config.start_frame)
        self.assertEqual(archive.config.outputs,config.outputs)
        self.assertEqual(archive.writer,writer)
        for executable in ('ffmpeg','bin/ffmpeg','./ffmpeg'):
            wire['writer_json'] = json.dumps(dict(asdict(writer),ffmpeg_bin=executable))
            self.assertEqual(archive_job(wire).writer.ffmpeg_bin,executable)
            self.assertFalse(archive_job(wire).writer_executable_absolute)

    def test_absolute_executable_preserved_and_same_flavour_required(self):
        wire = archival_wire()
        writer = asdict(WriterConfig(ffmpeg_bin=r'C:\tools\ffmpeg.exe'))
        wire['writer_json'] = json.dumps(writer)
        result = archive_job(wire)
        self.assertTrue(result.writer_executable_absolute)
        self.assertEqual(result.writer.ffmpeg_bin,r'C:\tools\ffmpeg.exe')
        wire['writer_json'] = json.dumps(dict(writer,ffmpeg_bin='/usr/bin/ffmpeg'))
        self.assertEqual(archive_job(wire).writer_executable_flavour,'posix')
        self.assertEqual(archive_job(wire).writer.ffmpeg_bin,'/usr/bin/ffmpeg')
        for executable in (r'C:ffmpeg',r'\\?\C:\ffmpeg.exe','https://example/ffmpeg'):
            wire['writer_json'] = json.dumps(dict(writer,ffmpeg_bin=executable))
            with self.assertRaises(ValueError):
                archive_job(wire)

    def test_mixed_archive_absolute_fields_refused(self):
        for field in ('output_dir','initial_hint_path','hint_video_path'):
            wire = archival_wire()
            wire[field] = '/foreign'
            with self.subTest(field=field),self.assertRaises(ValueError):
                archive_job(wire)
        for field in ('temp_dir','corrections_path'):
            wire = archival_wire()
            config = json.loads(wire['config_json'])
            config[field] = '/foreign'
            wire['config_json'] = json.dumps(config)
            with self.subTest(field=field),self.assertRaises(ValueError):
                archive_job(wire)

    def test_duplicate_nonfinite_embedded_and_outer_fields_refused(self):
        for field in ('config_json','writer_json'):
            for invalid in ('{"outputs":[],"outputs":[]}', '{"crf":NaN}', '{"crf":1e999}', '[]'):
                wire = archival_wire()
                wire[field] = invalid
                with self.subTest(field=field,invalid=invalid),self.assertRaises((ValueError,TypeError)):
                    archive_job(wire)
        raw = json.dumps(archival_wire())
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            archive_job(('{"profile_stages":false,'+raw[1:]).encode())

    def test_strict_types_unknown_fields_shared_setting_validation_and_sets(self):
        for value in (ArchivalJob,WriterConfig,WriterConfig(),{1,'mixed'}):
            invalid:Any = value
            with self.assertRaises(TypeError):
                archive_job(invalid)
        for field,value in [('profile_stages',1),('extra',None),('config_json',{}),('input_path',Path('/shots'))]:
            wire = archival_wire()
            wire[field] = value
            with self.subTest(field=field),self.assertRaises((ValueError,TypeError)):
                archive_job(wire)
        for field,value in [('outputs',['alpha',1]),('outputs',['alpha','alpha']),('outputs',{'alpha':True}),('model_scale',False),('model_scale',-1),('unknown',1)]:
            wire = archival_wire()
            config = json.loads(wire['config_json'])
            config[field] = value
            wire['config_json'] = json.dumps(config)
            with self.subTest(field=field,value=value),self.assertRaises((ValueError,TypeError)):
                archive_job(wire)
        wire = archival_wire()
        wire['extra'] = {1,'mixed'}
        with self.assertRaises(TypeError):
            archive_job(wire)

    def test_bounds_and_immutable_instance_contracts(self):
        with self.assertRaises(ValueError):
            archived_path('/'+('x'*MAX_PATH_BYTES),'posix')
        with self.assertRaises(ValueError):
            archive_job(b' '*(MAX_WIRE_BYTES+1))
        with self.assertRaises(TypeError):
            invalid:Any = bytearray(b'{}')
            ArchivalJob('windows',invalid)
        with self.assertRaises(ValueError):
            ArchivalJob('posix',json.dumps(archival_wire()).encode())

    def test_native_destination_lexical_guard_no_resolve(self):
        native = '/chosen/output' if os.name!='nt' else r'C:\chosen\output'
        foreign = r'C:\chosen\output' if os.name!='nt' else '/chosen/output'
        with patch.object(Path,'resolve',side_effect=AssertionError('no resolve')):
            self.assertEqual(str(native_destination(native)),native)
        for value in (foreign,'relative','/chosen/../output'):
            with self.assertRaisesRegex(ValueError,'Native destination requires an explicit absolute') as raised:
                native_destination(value)
            self.assertIsInstance(raised.exception.__cause__,ValueError)

    def test_existing_native_posix_literal_parents_preserved_collection_leaves_strict(self):
        from corridorkey.project_collection import safe_name
        for parent in ('parent\\literal','C:parent'):
            wire = archival_wire('posix')
            wire['input_path'] = '/shots/'+parent+'/frames'
            archived = archive_job(wire)
            self.assertEqual(archived.original_wire['input_path'],job_from_wire(wire).input_path)
            path = archived_path(wire['input_path'],'posix')
            self.assertEqual(path.parent.name,parent)
            self.assertEqual(safe_name(path.name),'frames')
            with self.assertRaises(ValueError):
                safe_name(parent)
