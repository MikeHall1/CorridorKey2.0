// Minimal official-suite test host, not a Resolve compatibility test.
#include "ofxImageEffect.h"
#include "ofxProperty.h"
#include "cache.hpp"
#include <cstdarg>
#include <functional>
#include <iostream>
#include <map>
#include <string>

extern "C" OfxPlugin* OfxGetPlugin(int);
namespace mock {
struct Properties {
    std::map<std::string,std::vector<int>> ints;
    std::map<std::string,std::vector<double>> doubles;
    std::map<std::string,std::vector<std::string>> strings;
    std::map<std::string,std::vector<void*>> pointers;
};
Properties& p(OfxPropertySetHandle h){return *reinterpret_cast<Properties*>(h);}
OfxPropertySetHandle handle(Properties& value){return reinterpret_cast<OfxPropertySetHandle>(&value);}
template<class T>OfxStatus set(std::map<std::string,std::vector<T>>& m,const char* key,int at,T value){if(at<0)return kOfxStatErrBadIndex;auto& v=m[key];v.resize(std::max(v.size(),std::size_t(at+1)));v[at]=value;return kOfxStatOK;}
template<class T>OfxStatus get(std::map<std::string,std::vector<T>>& m,const char* key,int at,T* out){auto it=m.find(key);if(it==m.end()||at<0||std::size_t(at)>=it->second.size())return kOfxStatErrUnknown;*out=it->second[at];return kOfxStatOK;}
OfxStatus si(OfxPropertySetHandle h,const char* key,int at,int v){return set(p(h).ints,key,at,v);}
OfxStatus sd(OfxPropertySetHandle h,const char* key,int at,double v){return set(p(h).doubles,key,at,v);}
OfxStatus ss(OfxPropertySetHandle h,const char* key,int at,const char* v){return set(p(h).strings,key,at,std::string(v));}
OfxStatus sp(OfxPropertySetHandle h,const char* key,int at,void* v){return set(p(h).pointers,key,at,v);}
OfxStatus gi(OfxPropertySetHandle h,const char* key,int at,int* v){return get(p(h).ints,key,at,v);}
OfxStatus gd(OfxPropertySetHandle h,const char* key,int at,double* v){return get(p(h).doubles,key,at,v);}
OfxStatus gs(OfxPropertySetHandle h,const char* key,int at,char** v){std::string s;auto r=get(p(h).strings,key,at,&s);if(r==kOfxStatOK)*v=const_cast<char*>(p(h).strings[key][at].c_str());return r;}
OfxStatus gp(OfxPropertySetHandle h,const char* key,int at,void** v){return get(p(h).pointers,key,at,v);}
OfxStatus gin(OfxPropertySetHandle h,const char* key,int n,int* v){for(int i=0;i<n;++i){auto r=gi(h,key,i,&v[i]);if(r!=kOfxStatOK)return r;}return kOfxStatOK;}
struct Param {Properties props;std::string kind,text;int number=0;};
struct Clip {Properties props,image;std::map<double,Properties> images;std::function<void()> onRead;};
struct Effect {Properties props;std::map<std::string,Param> params;std::map<std::string,Clip> clips;bool abort=false;};
int acquired=0,released=0;
OfxStatus effectProps(OfxImageEffectHandle e,OfxPropertySetHandle* out){*out=handle(reinterpret_cast<Effect*>(e)->props);return kOfxStatOK;}
OfxStatus paramSet(OfxImageEffectHandle e,OfxParamSetHandle* out){*out=reinterpret_cast<OfxParamSetHandle>(e);return kOfxStatOK;}
OfxStatus defineClip(OfxImageEffectHandle e,const char* name,OfxPropertySetHandle* out){*out=handle(reinterpret_cast<Effect*>(e)->clips[name].props);return kOfxStatOK;}
OfxStatus clipHandle(OfxImageEffectHandle e,const char* name,OfxImageClipHandle* out,OfxPropertySetHandle* prop){auto& clip=reinterpret_cast<Effect*>(e)->clips.at(name);*out=reinterpret_cast<OfxImageClipHandle>(&clip);if(prop)*prop=handle(clip.props);return kOfxStatOK;}
OfxStatus clipProps(OfxImageClipHandle c,OfxPropertySetHandle* out){*out=handle(reinterpret_cast<Clip*>(c)->props);return kOfxStatOK;}
OfxStatus image(OfxImageClipHandle c,OfxTime time,const OfxRectD*,OfxPropertySetHandle* out){auto& clip=*reinterpret_cast<Clip*>(c);auto onRead=clip.onRead;if(onRead)onRead();if(!clip.images.empty()){auto at=clip.images.find(time);if(at==clip.images.end())return kOfxStatFailed;*out=handle(at->second);}else *out=handle(clip.image);++acquired;return kOfxStatOK;}
OfxStatus release(OfxPropertySetHandle){++released;return kOfxStatOK;}
OfxStatus rod(OfxImageClipHandle c,OfxTime,OfxRectD* out){auto& props=reinterpret_cast<Clip*>(c)->props;*out={0,0,props.doubles["width"][0],props.doubles["height"][0]};return kOfxStatOK;}
int abort(OfxImageEffectHandle e){return reinterpret_cast<Effect*>(e)->abort?1:0;}
OfxStatus defineParam(OfxParamSetHandle set,const char* type,const char* name,OfxPropertySetHandle* out){auto& param=reinterpret_cast<Effect*>(set)->params[name];param.kind=type;*out=handle(param.props);return kOfxStatOK;}
OfxStatus paramHandle(OfxParamSetHandle set,const char* name,OfxParamHandle* out,OfxPropertySetHandle* prop){auto& param=reinterpret_cast<Effect*>(set)->params.at(name);*out=reinterpret_cast<OfxParamHandle>(&param);if(prop)*prop=handle(param.props);return kOfxStatOK;}
OfxStatus timedValue(OfxParamHandle h,OfxTime time,...){(void)time;auto& param=*reinterpret_cast<Param*>(h);va_list ap;va_start(ap,time);if(param.kind==kOfxParamTypeString)*va_arg(ap,char**)=const_cast<char*>(param.text.c_str());else *va_arg(ap,int*)=param.number;va_end(ap);return kOfxStatOK;}
OfxPropertySuiteV1 propSuite{};OfxImageEffectSuiteV1 effectSuite{};OfxParameterSuiteV1 paramSuite{};
const void* fetch(OfxPropertySetHandle,const char* name,int version){if(version!=1)return nullptr;if(std::strcmp(name,kOfxPropertySuite)==0)return &propSuite;if(std::strcmp(name,kOfxImageEffectSuite)==0)return &effectSuite;if(std::strcmp(name,kOfxParameterSuite)==0)return &paramSuite;return nullptr;}
void require(bool v,const char* message){if(!v)throw std::runtime_error(message);}
void setup(){propSuite.propSetInt=si;propSuite.propSetDouble=sd;propSuite.propSetString=ss;propSuite.propSetPointer=sp;propSuite.propGetInt=gi;propSuite.propGetDouble=gd;propSuite.propGetString=gs;propSuite.propGetPointer=gp;propSuite.propGetIntN=gin;
effectSuite.getPropertySet=effectProps;effectSuite.getParamSet=paramSet;effectSuite.clipDefine=defineClip;effectSuite.clipGetHandle=clipHandle;effectSuite.clipGetPropertySet=clipProps;effectSuite.clipGetImage=image;effectSuite.clipReleaseImage=release;effectSuite.clipGetRegionOfDefinition=rod;effectSuite.abort=abort;paramSuite.paramDefine=defineParam;paramSuite.paramGetHandle=paramHandle;paramSuite.paramGetValueAtTime=timedValue;}
}
int main(int argc,char** argv){
    using namespace mock;
    try{
        require(argc==3||argc==4,"host_test CACHE OUTPUT_PARENT [TOPDOWN_SELECTED_SOURCE_RGB_F32_LE]");setup();OfxHost host{nullptr,fetch};auto* plugin=OfxGetPlugin(0);plugin->setHost(&host);
        require(plugin->mainEntry(kOfxActionLoad,nullptr,nullptr,nullptr)==kOfxStatOK,"load");
        auto frame=ck::frame(std::filesystem::u8path(argv[1]),0);
        std::vector<std::uint8_t> suppliedSource;
        if(argc==4){
            require(frame.bound,"external source fixture requires bound cache");
            const auto size=std::uint64_t(frame.width)*frame.height*frame.count*12;
            // This optional input is a tiny test fixture, not a product reader.
            require(size<=64*1024*1024,"external test source exceeds64MiB fixture cap");
            std::ifstream stream(std::filesystem::u8path(argv[3]),std::ios::binary);require(bool(stream),"external source fixture missing");
            stream.seekg(0,std::ios::end);require(stream.tellg()==static_cast<std::streamoff>(size),"external source fixture size");stream.seekg(0);
            suppliedSource.resize(static_cast<std::size_t>(size));ck::read(stream,suppliedSource.data(),suppliedSource.size());
        }
        for(int instanceIndex=0;instanceIndex<2;++instanceIndex){
            Effect e;auto h=reinterpret_cast<OfxImageEffectHandle>(&e);
            auto call=[&](const char* action,Properties* args=nullptr,Properties* out=nullptr){return plugin->mainEntry(action,h,args?handle(*args):nullptr,out?handle(*out):nullptr);};
            require(call(kOfxActionDescribe)==kOfxStatOK,"describe");require(call(kOfxImageEffectActionDescribeInContext)==kOfxStatOK,"context");require(call(kOfxActionCreateInstance)==kOfxStatOK,"create");
            e.params["CacheFile"].text=argv[1];e.params["CacheStart"].number=10+instanceIndex*10;
            auto& source=e.clips["Source"].props;source.doubles[kOfxImagePropPixelAspectRatio]={1};source.doubles[kOfxImageEffectPropFrameRate]={frame.fps};source.doubles["width"]={double(frame.width)};source.doubles["height"]={double(frame.height)};
            Properties prefs;require(call(kOfxImageEffectActionGetClipPreferences,nullptr,&prefs)==kOfxStatOK,"preferences");require(prefs.strings[kOfxImageEffectPropPreMultiplication][0]==kOfxImagePreMultiplied,"association preference");
            int stride=static_cast<int>(frame.width*16+32);std::vector<std::uint8_t> buffer(std::size_t(stride)*frame.height,0x7f);
            if(instanceIndex)stride=-stride;auto* base=buffer.data()+(instanceIndex?std::size_t(-stride)*(frame.height-1):0);
            auto& output=e.clips["Output"].image;output.strings[kOfxImageEffectPropPixelDepth]={kOfxBitDepthFloat};output.strings[kOfxImageEffectPropComponents]={kOfxImageComponentRGBA};output.strings[kOfxImagePropField]={kOfxImageFieldNone};output.strings[kOfxImageEffectPropPreMultiplication]={kOfxImagePreMultiplied};output.doubles[kOfxImagePropPixelAspectRatio]={1};output.ints[kOfxImagePropBounds]={0,0,int(frame.width),int(frame.height)};output.ints[kOfxImagePropRowBytes]={stride};output.pointers[kOfxImagePropData]={base};
            std::vector<std::vector<std::uint8_t>> sourceBuffers;
            if(frame.bound){
                sourceBuffers.resize(frame.count);
                for(std::uint32_t at=0;at<frame.count;++at){
                    auto& bytes=sourceBuffers[at];bytes.resize(buffer.size(),0x2b);
                    auto* sourceBase=bytes.data()+(instanceIndex?std::size_t(-stride)*(frame.height-1):0);
                    for(std::uint32_t top=0;top<frame.height;++top)for(std::uint32_t x=0;x<frame.width;++x){
                        float rgba[4]={float(at+1)/4+float(x)/16,float(top+1)/8,-.5f,1.f};
                        if(!suppliedSource.empty()){const auto offset=((std::size_t(at)*frame.height+top)*frame.width+x)*12;for(int channel=0;channel<3;++channel)rgba[channel]=ck::scalar(suppliedSource.data()+offset+channel*4);}
                        std::memcpy(sourceBase+std::ptrdiff_t(frame.height-1-top)*stride+std::size_t(x)*16,rgba,16);
                    }
                    auto& sourceImage=e.clips["Source"].images[double(e.params["CacheStart"].number)+at];sourceImage=output;
                    sourceImage.doubles[kOfxImageEffectPropRenderScale]={1,1};sourceImage.pointers[kOfxImagePropData]={sourceBase};
                }
            }
            Properties args;args.doubles[kOfxImageEffectPropRenderScale]={1,1};args.strings[kOfxImageEffectPropFieldToRender]={kOfxImageFieldNone};args.ints[kOfxImageEffectPropRenderWindow]={0,0,int(frame.width),int(frame.height)};
            if(frame.bound){Properties needed;args.doubles[kOfxPropTime]={double(e.params["CacheStart"].number)};require(call(kOfxImageEffectActionGetFramesNeeded,&args,&needed)==kOfxStatOK,"bound dependency range");const auto& range=needed.doubles["OfxImageClipPropFrameRange_Source"];require(range.size()==2&&range[0]==e.params["CacheStart"].number&&range[1]==e.params["CacheStart"].number+frame.count-1,"incomplete source dependency range");}
            for(unsigned at:{frame.count-1,0u,frame.count/2,0u}){
                args.doubles[kOfxPropTime]={double(e.params["CacheStart"].number)+at};require(call(kOfxImageEffectActionRender,&args)==kOfxStatOK,"render");
                std::ofstream out(std::filesystem::u8path(argv[2])/std::filesystem::path(std::to_string(instanceIndex)+"-"+std::to_string(at)+".rgba"),std::ios::binary);
                for(int y=int(frame.height)-1;y>=0;--y)out.write(reinterpret_cast<char*>(base+std::ptrdiff_t(y)*stride),frame.width*16);
            }
            auto full=buffer;std::fill(buffer.begin(),buffer.end(),0x7f);args.ints[kOfxImageEffectPropRenderWindow]={1,1,int(frame.width)-1,int(frame.height)-1};
            require(call(kOfxImageEffectActionRender,&args)==kOfxStatOK,"cropped window");
            for(unsigned y=0;y<frame.height;++y)for(unsigned x=0;x<frame.width;++x){
                auto offset=(base-buffer.data())+std::ptrdiff_t(y)*stride+std::ptrdiff_t(x)*16;
                const bool inside=x>=1&&x<frame.width-1&&y>=1&&y<frame.height-1;
                for(int c=0;c<16;++c)require(buffer[std::size_t(offset)+c]==(inside?full[std::size_t(offset)+c]:0x7f),"window overwrite/mismatch");
            }
            args.ints[kOfxImageEffectPropRenderWindow]={0,0,int(frame.width),int(frame.height)};
            if(frame.bound){
                args.doubles[kOfxPropTime]={double(e.params["CacheStart"].number)+frame.count-1};
                for(std::uint32_t changed:{0u,frame.count-1}){
                    auto& sourceImage=e.clips["Source"].images.at(double(e.params["CacheStart"].number)+changed);
                    auto* pixels=static_cast<std::uint8_t*>(sourceImage.pointers[kOfxImagePropData][0]);
                    std::uint8_t previous[16];std::memcpy(previous,pixels,16);
                    for(int channel:{0,1,3}){
                        const float wrong=channel==3?.5f:(channel==1?std::numeric_limits<float>::quiet_NaN():9.f);std::memcpy(pixels+channel*4,&wrong,4);std::fill(buffer.begin(),buffer.end(),0x7f);
                        require(call(kOfxImageEffectActionRender,&args)!=kOfxStatOK,"changed source pixels accepted");require(std::all_of(buffer.begin(),buffer.end(),[](std::uint8_t value){return value==0x7f;}),"output written before source validation");std::memcpy(pixels,previous,16);
                    }
                    const auto saved=sourceImage;sourceImage.doubles[kOfxImageEffectPropRenderScale]={.5,.5};require(call(kOfxImageEffectActionRender,&args)!=kOfxStatOK,"source proxy accepted");sourceImage=saved;
                    sourceImage.strings[kOfxImagePropField]={kOfxImageFieldUpper};require(call(kOfxImageEffectActionRender,&args)!=kOfxStatOK,"source field accepted");sourceImage=saved;
                    sourceImage.doubles[kOfxImagePropPixelAspectRatio]={2};require(call(kOfxImageEffectActionRender,&args)!=kOfxStatOK,"source PAR accepted");sourceImage=saved;
                    sourceImage.ints[kOfxImagePropRowBytes]={1};require(call(kOfxImageEffectActionRender,&args)!=kOfxStatOK,"short source stride accepted");sourceImage=saved;
                }
                const auto mapped=e.params["CacheStart"].number;e.params["CacheStart"].number=mapped+1;
                require(call(kOfxImageEffectActionRender,&args)!=kOfxStatOK,"changed time/source mapping accepted");e.params["CacheStart"].number=mapped;
                require(call(kOfxImageEffectActionRender,&args)==kOfxStatOK,"source restoration failed");
                // A source request may synchronously evaluate another instance
                // of this plugin upstream. A global render lock deadlocks here.
                Effect upstream=e;auto upstreamHandle=reinterpret_cast<OfxImageEffectHandle>(&upstream);
                std::vector<std::uint8_t> upstreamBuffer(buffer.size(),0x7f);
                upstream.clips["Output"].image.pointers[kOfxImagePropData]={upstreamBuffer.data()+(instanceIndex?std::size_t(-stride)*(frame.height-1):0)};
                require(plugin->mainEntry(kOfxActionCreateInstance,upstreamHandle,nullptr,nullptr)==kOfxStatOK,"upstream create");
                bool evaluated=false;e.clips["Source"].onRead=[&](){e.clips["Source"].onRead=nullptr;evaluated=true;require(plugin->mainEntry(kOfxImageEffectActionRender,upstreamHandle,handle(args),nullptr)==kOfxStatOK,"upstream source evaluation");};
                require(call(kOfxImageEffectActionRender,&args)==kOfxStatOK&&evaluated,"nested instance render");
                require(plugin->mainEntry(kOfxActionDestroyInstance,upstreamHandle,nullptr,nullptr)==kOfxStatOK,"upstream destroy");
            }
            output.ints[kOfxImagePropRowBytes]={1};require(call(kOfxImageEffectActionRender,&args)!=kOfxStatOK,"short stride accepted");output.ints[kOfxImagePropRowBytes]={stride};
            output.strings[kOfxImagePropField]={kOfxImageFieldLower};require(call(kOfxImageEffectActionRender,&args)!=kOfxStatOK,"fielded image accepted");output.strings[kOfxImagePropField]={kOfxImageFieldNone};
            output.doubles[kOfxImagePropPixelAspectRatio]={2};require(call(kOfxImageEffectActionRender,&args)!=kOfxStatOK,"output PAR accepted");output.doubles[kOfxImagePropPixelAspectRatio]={1};
            auto goodTime=args.doubles[kOfxPropTime];args.doubles[kOfxPropTime][0]+=.5;require(call(kOfxImageEffectActionRender,&args)!=kOfxStatOK,"fractional time accepted");args.doubles[kOfxPropTime]=goodTime;
            output.strings[kOfxImageEffectPropPreMultiplication]={kOfxImageUnPreMultiplied};require(call(kOfxImageEffectActionRender,&args)!=kOfxStatOK,"wrong association accepted");output.strings[kOfxImageEffectPropPreMultiplication]={kOfxImagePreMultiplied};
            args.doubles[kOfxImageEffectPropRenderScale]={.5,.5};require(call(kOfxImageEffectActionRender,&args)!=kOfxStatOK,"proxy accepted");args.doubles[kOfxImageEffectPropRenderScale]={1,1};
            e.abort=true;require(call(kOfxImageEffectActionRender,&args)!=kOfxStatOK,"abort accepted");e.abort=false;
            e.params["CacheFile"].text="nonexistent.ckcache";require(call(kOfxImageEffectActionRender,&args)!=kOfxStatOK,"missing cache accepted");
            require(call(kOfxActionDestroyInstance)==kOfxStatOK,"destroy");
        }
        require(acquired==released,"host image leak");require(plugin->mainEntry(kOfxActionUnload,nullptr,nullptr,nullptr)==kOfxStatOK,"unload");std::cout<<"host ABI tests passed; image leases="<<acquired<<"\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
    return 0;
}
