// Explicit upstream capture staging. No host calls, inference or cache mutation.
#pragma once
#include "cache.hpp"
#include <cerrno>
#include <iomanip>
#include <memory>
#include <locale>
#include <random>
#include <sstream>
#include <system_error>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ck::capture {
constexpr std::uint64_t maxBytes=512ull*1024*1024;
constexpr std::uint64_t maxPixels=16ull*1024*1024;
constexpr std::uint32_t maxFrames=120;
struct Cancelled:std::runtime_error{Cancelled():std::runtime_error("Upstream capture cancelled") {}};
inline std::string hex(const Digest& digest){static constexpr char digits[]="0123456789abcdef";std::string out;out.reserve(64);for(auto value:digest){out+=digits[value>>4];out+=digits[value&15];}return out;}
inline bool alias(const std::filesystem::path& path){
#if defined(_WIN32)
    const auto value=GetFileAttributesW(path.c_str());
    if(value==INVALID_FILE_ATTRIBUTES)throw std::runtime_error("Capture path metadata unavailable");
    return (value&FILE_ATTRIBUTE_REPARSE_POINT)!=0;
#else
    return std::filesystem::is_symlink(std::filesystem::symlink_status(path));
#endif
}
inline void plainParents(const std::filesystem::path& path){
    if(!path.is_absolute())throw std::runtime_error("Capture parent must be an explicit absolute native directory");
    for(const auto& part:path)if(part==".."||part==".")throw std::runtime_error("Capture path traversal is unsupported");
    for(auto at=path;;at=at.parent_path()){
        if(alias(at))throw std::runtime_error("Capture aliases/reparse paths are unsupported");
        const auto parent=at.parent_path();if(parent.empty()||parent==at)break;
    }
}
struct DirectoryIdentity{std::uint64_t device=0,inode=0;};
inline DirectoryIdentity directoryIdentity(const std::filesystem::path& path){
    plainParents(path);if(!std::filesystem::is_directory(path))throw std::runtime_error("Capture directory unavailable");
#if defined(_WIN32)
    HANDLE handle=CreateFileW(path.c_str(),FILE_READ_ATTRIBUTES,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,nullptr,OPEN_EXISTING,FILE_FLAG_BACKUP_SEMANTICS|FILE_FLAG_OPEN_REPARSE_POINT,nullptr);
    if(handle==INVALID_HANDLE_VALUE)throw std::runtime_error("Cannot pin capture directory");
    BY_HANDLE_FILE_INFORMATION info{};const auto ok=GetFileInformationByHandle(handle,&info);CloseHandle(handle);
    if(!ok||(info.dwFileAttributes&FILE_ATTRIBUTE_REPARSE_POINT))throw std::runtime_error("Cannot verify capture directory identity");
    return {info.dwVolumeSerialNumber,(std::uint64_t(info.nFileIndexHigh)<<32)|info.nFileIndexLow};
#else
    struct stat info{};if(::lstat(path.c_str(),&info)||!S_ISDIR(info.st_mode))throw std::runtime_error("Cannot pin capture directory");
    return {std::uint64_t(info.st_dev),std::uint64_t(info.st_ino)};
#endif
}
class File {
#if defined(_WIN32)
    HANDLE handle=INVALID_HANDLE_VALUE;
#else
    int handle=-1;
#endif
public:
    File()=default;File(const File&)=delete;File& operator=(const File&)=delete;
    ~File(){close();}
    void open(const std::filesystem::path& path){
#if defined(_WIN32)
        handle=CreateFileW(path.c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL|FILE_FLAG_OPEN_REPARSE_POINT,nullptr);
        if(handle==INVALID_HANDLE_VALUE)throw std::runtime_error("Cannot exclusively create capture file");
#else
        handle=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
        if(handle<0)throw std::runtime_error("Cannot exclusively create capture file");
#endif
    }
    void write(const std::uint8_t* data,std::size_t size){
        while(size){
#if defined(_WIN32)
            DWORD amount=0;const auto bounded=static_cast<DWORD>(std::min(size,std::size_t(1024*1024)));
            if(!WriteFile(handle,data,bounded,&amount,nullptr)||!amount)throw std::runtime_error("Capture write failed");
#else
            const auto amount=::write(handle,data,size);if(amount<0&&errno==EINTR)continue;
            if(amount<=0)throw std::runtime_error("Capture write failed");
#endif
            data+=amount;size-=static_cast<std::size_t>(amount);
        }
    }
    void sync(){
#if defined(_WIN32)
        if(!FlushFileBuffers(handle))throw std::runtime_error("Capture file sync failed");
#else
        if(::fsync(handle))throw std::runtime_error("Capture file sync failed");
#endif
    }
    void closeChecked(){
#if defined(_WIN32)
        if(handle!=INVALID_HANDLE_VALUE){const auto previous=handle;handle=INVALID_HANDLE_VALUE;if(!CloseHandle(previous))throw std::runtime_error("Capture file close failed");}
#else
        if(handle>=0){const auto previous=handle;handle=-1;if(::close(previous))throw std::runtime_error("Capture file close failed");}
#endif
    }
    void close() noexcept {
#if defined(_WIN32)
        if(handle!=INVALID_HANDLE_VALUE){CloseHandle(handle);handle=INVALID_HANDLE_VALUE;}
#else
        if(handle>=0){::close(handle);handle=-1;}
#endif
    }
};
struct Spec {
    std::filesystem::path parent;
    std::int32_t hostStart=0;
    std::uint32_t count=0,width=0,height=0;
    double fps=0;
    std::uint64_t budget=maxBytes;
    std::uint64_t bytes()const{return std::uint64_t(count)*width*height*12;}
    void validate()const{
        const auto pixels=std::uint64_t(width)*height;
        if(!count||count>maxFrames||!width||!height||width>32768||height>32768||pixels>maxPixels||!std::isfinite(fps)||fps<=0||!budget||budget>maxBytes||bytes()>budget)
            throw std::runtime_error("Capture range/geometry/FPS/byte budget unsupported (max120 frames,16M pixels,512MiB)");
        if(std::int64_t(hostStart)+count-1>INT32_MAX)throw std::runtime_error("Capture host frame range overflows signed32 mapping");
    }
};
class Bundle {
    Spec spec;
    std::filesystem::path root,manifest;
    std::string manifestUtf8;
    DirectoryIdentity owner{};
    std::string id;
    File payload;
    Sha256 payloadHash;
    std::vector<Digest> hashes;
    bool created=false,committed=false,payloadOwned=false,metadataOwned=false;
public:
    explicit Bundle(const Spec& value):spec(value){
        spec.validate();directoryIdentity(spec.parent);
        try{
            std::random_device random;static constexpr char digits[]="0123456789abcdef";
            for(int attempt=0;attempt<32;++attempt){
                id.clear();for(int byte=0;byte<16;++byte){const auto value=static_cast<unsigned>(random())&255u;id+=digits[value>>4];id+=digits[value&15];}
                root=spec.parent/std::filesystem::u8path("corridorkey-capture-"+id);
                if(std::filesystem::create_directory(root)){created=true;break;}
            }
            if(!created)throw std::runtime_error("Cannot reserve a fresh exclusive capture directory");
#if !defined(_WIN32)
            if(::chmod(root.c_str(),0700))throw std::runtime_error("Cannot restrict capture directory permissions");
#endif
            owner=directoryIdentity(root);manifest=root/"capture.json";manifestUtf8=manifest.u8string();payload.open(root/"frames.rgb32");payloadOwned=true;hashes.reserve(spec.count);
        }catch(const std::exception& error){const std::string message=error.what();const auto note=cleanup();throw std::runtime_error(message+note);}
    }
    ~Bundle(){cleanup();}
    const std::filesystem::path& manifestPath()const{return manifest;}
    const std::string& manifestText()const{return manifestUtf8;}
    bool isCommitted()const{return committed;}
    std::string cleanup()noexcept{
        payload.close();if(!created||committed)return {};
        try{
            const auto current=directoryIdentity(root);
            if(current.device!=owner.device||current.inode!=owner.inode)throw std::runtime_error("capture directory identity changed");
            for(const auto& item:std::array<std::pair<const char*,bool>,2>{{{"frames.rgb32",payloadOwned},{"capture.json",metadataOwned}}}){
                if(!item.second)continue;const auto file=root/item.first;
                if(std::filesystem::exists(std::filesystem::symlink_status(file)))std::filesystem::remove(file);
            }
            if(!std::filesystem::remove(root))throw std::runtime_error("capture directory is not empty");
            created=false;return {};
        }catch(...){return "; cleanup NOT VERIFIED; retained capture path: "+root.u8string();}
    }
    template<class Reader,class Check>void frame(std::uint32_t index,Reader reader,Check check){
        if(index!=hashes.size()||index>=spec.count)throw std::runtime_error("Capture frames must be complete and chronological");
        Sha256 frameHash;static constexpr char domain[]="CorridorKey/source/linear-rgb-f32/v1\0";
        frameHash.add(reinterpret_cast<const std::uint8_t*>(domain),sizeof(domain)-1);
        std::uint8_t metadata[12]{};const std::uint32_t values[3]={index,spec.width,spec.height};
        for(int field=0;field<3;++field)for(int byte=0;byte<4;++byte)metadata[field*4+byte]=static_cast<std::uint8_t>(values[field]>>(byte*8));
        frameHash.add(metadata,sizeof(metadata));
        const auto capacity=std::min(spec.width,4096u);std::vector<std::uint8_t> chunk(std::size_t(capacity)*12);
        for(std::uint32_t y=0;y<spec.height;++y)for(std::uint32_t x=0;x<spec.width;x+=capacity){
            check((double(index)+(double(y)*spec.width+x)/(double(spec.width)*spec.height))/spec.count);
            const auto count=std::min(capacity,spec.width-x);reader(y,x,count,chunk.data());
            const auto size=std::size_t(count)*12;frameHash.add(chunk.data(),size);payloadHash.add(chunk.data(),size);payload.write(chunk.data(),size);
        }
        hashes.push_back(frameHash.finish());
    }
    void commit(){
        if(committed||hashes.size()!=spec.count)throw std::runtime_error("Capture cannot publish incomplete/repeated frames");
        payload.sync();payload.closeChecked();
        std::ostringstream json;json.imbue(std::locale::classic());json<<std::setprecision(std::numeric_limits<double>::max_digits10)
          <<"{\"schema_version\":1,\"type\":\"corridorkey_ofx_capture\",\"capture_id\":\""<<id
          <<"\",\"source_scope\":\"ofx_instance_upstream\",\"host_start\":"<<spec.hostStart
          <<",\"frame_count\":"<<spec.count<<",\"width\":"<<spec.width<<",\"height\":"<<spec.height
          <<",\"fps\":"<<spec.fps<<",\"pixel_aspect\":1.0,\"colour_contract\":\"scene_linear_bt709_d65_opaque_rgb\",\"payload\":{\"path\":\"frames.rgb32\",\"bytes\":"
          <<spec.bytes()<<",\"sha256\":\""<<hex(payloadHash.finish())<<"\"},\"source_digests\":[";
        for(std::size_t index=0;index<hashes.size();++index){if(index)json<<',';json<<'"'<<hex(hashes[index])<<'"';}json<<"]}\n";
        const auto raw=json.str();File metadata;metadata.open(manifest);metadataOwned=true;metadata.write(reinterpret_cast<const std::uint8_t*>(raw.data()),raw.size());metadata.sync();metadata.closeChecked();
        const auto current=directoryIdentity(root);if(current.device!=owner.device||current.inode!=owner.inode)throw std::runtime_error("Capture directory changed before commit");
        File marker;marker.open(root/"COMPLETE");committed=true;marker.close();
        // Marker creation is the commit point. No fallible sync/callback follows;
        // directory-entry survival across power loss is not claimed.
    }
};
} // namespace ck::capture
