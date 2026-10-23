// Experimental cached-frame OFX adapter; inference stays in the shared engine.
#if defined(_WIN32) && (defined(_M_ARM64EC) || (!defined(_M_X64) && !defined(__x86_64__)))
#error "This experimental Windows bundle targets Intel x64 only."
#endif
#include "ofxImageEffect.h"
#include "ofxProperty.h"
#include "cache.hpp"
#include <memory>
#include <mutex>
#include <string>

namespace {
OfxHost* host=nullptr;
OfxImageEffectSuiteV1* effects=nullptr;
OfxPropertySuiteV1* props=nullptr;
OfxParameterSuiteV1* params=nullptr;
OfxMessageSuiteV1* messages=nullptr;
struct Instance {OfxImageClipHandle output=nullptr,source=nullptr;OfxParamHandle file=nullptr,start=nullptr;std::mutex rendering;};
void check(OfxStatus s){if(s!=kOfxStatOK)throw std::runtime_error("Required host operation/property unavailable");}
OfxPropertySetHandle properties(OfxImageEffectHandle h){OfxPropertySetHandle p=nullptr;check(effects->getPropertySet(h,&p));return p;}
Instance* instance(OfxImageEffectHandle h){void* p=nullptr;check(props->propGetPointer(properties(h),kOfxPropInstanceData,0,&p));if(!p)throw std::runtime_error("Missing instance");return static_cast<Instance*>(p);}
void text(OfxPropertySetHandle p,const char* name,const char* value){check(props->propSetString(p,name,0,value));}
void integer(OfxPropertySetHandle p,const char* name,int value){check(props->propSetInt(p,name,0,value));}
double number(OfxPropertySetHandle p,const char* name,int index=0){double value=0;check(props->propGetDouble(p,name,index,&value));return value;}
std::string string(OfxPropertySetHandle p,const char* name){char* value=nullptr;check(props->propGetString(p,name,0,&value));return value?value:"";}
void describe(OfxImageEffectHandle h){
    auto p=properties(h);
    text(p,kOfxPropLabel,"CorridorKey Cached (experimental)");
    text(p,kOfxPropPluginDescription,"Reads explicitly prepared linear RGBA cache. No live inference; missing frames fail. Host/source colour matching must be validated.");
    text(p,kOfxImageEffectPropSupportedContexts,kOfxImageEffectContextFilter);
    text(p,kOfxImageEffectPropSupportedPixelDepths,kOfxBitDepthFloat);
    text(p,kOfxImageEffectPluginRenderThreadSafety,kOfxImageEffectRenderInstanceSafe);
    integer(p,kOfxImageEffectPluginPropHostFrameThreading,0);
    integer(p,kOfxImageEffectPropSupportsTiles,0);
    integer(p,kOfxImageEffectPropSupportsMultiResolution,0);
    integer(p,kOfxImageEffectPropTemporalClipAccess,1);
}
void context(OfxImageEffectHandle h){
    for(const char* name:{"Source","Output"}){
        OfxPropertySetHandle p=nullptr;check(effects->clipDefine(h,name,&p));
        text(p,kOfxImageEffectPropSupportedComponents,kOfxImageComponentRGBA);
        integer(p,kOfxImageEffectPropSupportsTiles,0);
    }
    OfxParamSetHandle set=nullptr;check(effects->getParamSet(h,&set));OfxPropertySetHandle p=nullptr;
    check(params->paramDefine(set,kOfxParamTypeString,"CacheFile",&p));
    text(p,kOfxPropLabel,"Prepared cache file");text(p,kOfxParamPropStringMode,kOfxParamStringIsFilePath);
    text(p,kOfxParamPropDefault,"");integer(p,kOfxParamPropAnimates,0);
    check(params->paramDefine(set,kOfxParamTypeInteger,"CacheStart",&p));
    text(p,kOfxPropLabel,"Host frame for cache frame 0");integer(p,kOfxParamPropDefault,0);integer(p,kOfxParamPropAnimates,0);
}
void create(OfxImageEffectHandle h){
    auto p=std::make_unique<Instance>();OfxParamSetHandle set=nullptr;
    check(effects->clipGetHandle(h,"Output",&p->output,nullptr));check(effects->clipGetHandle(h,"Source",&p->source,nullptr));
    check(effects->getParamSet(h,&set));check(params->paramGetHandle(set,"CacheFile",&p->file,nullptr));check(params->paramGetHandle(set,"CacheStart",&p->start,nullptr));
    check(props->propSetPointer(properties(h),kOfxPropInstanceData,0,p.get()));p.release();
}
struct Image {
    OfxPropertySetHandle handle=nullptr;
    ~Image(){if(handle)effects->clipReleaseImage(handle);}
};
// Bound caches conservatively depend on every selected source frame. No verified
// source state is retained between renders: host notification is not an epoch.
bool verifySource(OfxImageEffectHandle h,Instance& state,const ck::Frame& frame,int start){
    for(std::uint32_t index=0;index<frame.count;++index){
        if(effects->abort(h))return false;
        const double time=double(start)+index;
        OfxRectD rod{};check(effects->clipGetRegionOfDefinition(state.source,time,&rod));
        if(rod.x1!=0||rod.y1!=0||rod.x2!=frame.width||rod.y2!=frame.height)throw std::runtime_error("Bound source geometry/origin changed");
        Image image;check(effects->clipGetImage(state.source,time,nullptr,&image.handle));
        const auto association=string(image.handle,kOfxImageEffectPropPreMultiplication);
        if(string(image.handle,kOfxImageEffectPropPixelDepth)!=kOfxBitDepthFloat||string(image.handle,kOfxImageEffectPropComponents)!=kOfxImageComponentRGBA||
           string(image.handle,kOfxImagePropField)!=kOfxImageFieldNone||number(image.handle,kOfxImagePropPixelAspectRatio)!=1||
           number(image.handle,kOfxImageEffectPropRenderScale,0)!=1||number(image.handle,kOfxImageEffectPropRenderScale,1)!=1||
           (association!=kOfxImageOpaque&&association!=kOfxImageUnPreMultiplied&&association!=kOfxImagePreMultiplied))
            throw std::runtime_error("Bound source requires native unfielded square-pixel float RGBA, opaque scene-linear BT.709 RGB");
        int bounds[4]{},stride=0;void* data=nullptr;
        check(props->propGetIntN(image.handle,kOfxImagePropBounds,4,bounds));check(props->propGetInt(image.handle,kOfxImagePropRowBytes,0,&stride));
        check(props->propGetPointer(image.handle,kOfxImagePropData,0,&data));
        const auto magnitude=stride<0?-std::int64_t(stride):std::int64_t(stride);
        if(!data||bounds[0]!=0||bounds[1]!=0||bounds[2]!=static_cast<int>(frame.width)||bounds[3]!=static_cast<int>(frame.height)||
           magnitude<std::int64_t(frame.width)*16||std::uint64_t(magnitude)*(frame.height-1)+frame.width*16>std::uint64_t(PTRDIFF_MAX))
            throw std::runtime_error("Unsupported bound source bounds/stride");
        ck::Sha256 hash;static constexpr char domain[]="CorridorKey/source/linear-rgb-f32/v1\0";
        hash.add(reinterpret_cast<const std::uint8_t*>(domain),sizeof(domain)-1);
        std::uint8_t metadata[12]{};const std::uint32_t fields[3]={index,frame.width,frame.height};
        for(int field=0;field<3;++field)for(int byte=0;byte<4;++byte)metadata[field*4+byte]=static_cast<std::uint8_t>(fields[field]>>(byte*8));
        hash.add(metadata,sizeof(metadata));
        const auto chunkPixels=std::min(frame.width,4096u);std::vector<std::uint8_t> rgb(std::size_t(chunkPixels)*12);
        for(std::uint32_t top=0;top<frame.height;++top){
            if(effects->abort(h))return false;
            const auto* row=static_cast<const std::uint8_t*>(data)+std::ptrdiff_t(frame.height-1-top)*stride;
            for(std::uint32_t x=0;x<frame.width;++x){
                float rgba[4];std::memcpy(rgba,row+std::size_t(x)*16,16);
                if(rgba[3]!=1.f)throw std::runtime_error("Bound source alpha must be exactly opaque; alpha conversion unsupported");
                for(int channel=0;channel<3;++channel){
                    if(!std::isfinite(rgba[channel]))throw std::runtime_error("Nonfinite bound source RGB");
                    std::uint32_t bits;std::memcpy(&bits,&rgba[channel],4);
                    for(int byte=0;byte<4;++byte)rgb[std::size_t(x%chunkPixels)*12+channel*4+byte]=static_cast<std::uint8_t>(bits>>(byte*8));
                }
                if((x+1)%chunkPixels==0||x+1==frame.width)hash.add(rgb.data(),std::size_t(x%chunkPixels+1)*12);
            }
        }
        if(hash.finish()!=frame.sourceDigests[index])throw std::runtime_error("Bound source pixels differ from producer witness; prepare a new render/cache or correct source/time/colour mapping");
    }
    return true;
}
OfxStatus framesNeeded(OfxImageEffectHandle h,OfxPropertySetHandle args,OfxPropertySetHandle out){
    auto* state=instance(h);const double time=number(args,kOfxPropTime);char* name=nullptr;int start=0;
    check(params->paramGetValueAtTime(state->file,time,&name));check(params->paramGetValueAtTime(state->start,time,&start));
    if(!name||!*name)return kOfxStatReplyDefault;
    const auto frame=ck::frame(std::filesystem::u8path(name),0);
    if(!frame.bound)return kOfxStatReplyDefault;
    check(props->propSetDouble(out,"OfxImageClipPropFrameRange_Source",0,double(start)));
    check(props->propSetDouble(out,"OfxImageClipPropFrameRange_Source",1,double(start)+frame.count-1));
    return kOfxStatOK;
}
OfxStatus render(OfxImageEffectHandle h,OfxPropertySetHandle args){
    auto* state=instance(h);std::lock_guard<std::mutex> lock(state->rendering);double time=number(args,kOfxPropTime);
    if(!std::isfinite(time)||number(args,kOfxImageEffectPropRenderScale,0)!=1||number(args,kOfxImageEffectPropRenderScale,1)!=1)
        throw std::runtime_error("Only native scale and finite integer frame times supported");
    if(string(args,kOfxImageEffectPropFieldToRender)!=kOfxImageFieldNone)throw std::runtime_error("Fielded rendering unsupported");
    char* name=nullptr;int start=0;
    check(params->paramGetValueAtTime(state->file,time,&name));check(params->paramGetValueAtTime(state->start,time,&start));
    const double at=time-start;
    if(!name||!*name||at<0||at>999999||std::abs(at-std::round(at))>1e-6)throw std::runtime_error("Select prepared cache and exact in-range frame offset; retiming interpolation unsupported");
    if(effects->abort(h))return kOfxStatFailed;
    const auto frame=ck::frame(std::filesystem::u8path(name),static_cast<std::uint32_t>(std::llround(at)));
    OfxPropertySetHandle sourceProps=nullptr;check(effects->clipGetPropertySet(state->source,&sourceProps));
    if(number(sourceProps,kOfxImagePropPixelAspectRatio)!=1||!std::isfinite(number(sourceProps,kOfxImageEffectPropFrameRate))||
       std::abs(number(sourceProps,kOfxImageEffectPropFrameRate)-frame.fps)>.001)throw std::runtime_error("Source PAR/FPS differs from cache");
    OfxRectD rod{};check(effects->clipGetRegionOfDefinition(state->source,time,&rod));
    if(rod.x1!=0||rod.y1!=0||rod.x2!=frame.width||rod.y2!=frame.height)throw std::runtime_error("Source geometry/origin differs from cache");
    if(frame.bound&&!verifySource(h,*state,frame,start))return kOfxStatFailed;
    Image image;check(effects->clipGetImage(state->output,time,nullptr,&image.handle));
    if(string(image.handle,kOfxImageEffectPropPixelDepth)!=kOfxBitDepthFloat||string(image.handle,kOfxImageEffectPropComponents)!=kOfxImageComponentRGBA||
       string(image.handle,kOfxImagePropField)!=kOfxImageFieldNone||number(image.handle,kOfxImagePropPixelAspectRatio)!=1||
       string(image.handle,kOfxImageEffectPropPreMultiplication)!=kOfxImagePreMultiplied)
        throw std::runtime_error("Unfielded square-pixel premultiplied float RGBA output required");
    int bounds[4]{},window[4]{},stride=0;void* data=nullptr;
    check(props->propGetIntN(image.handle,kOfxImagePropBounds,4,bounds));check(props->propGetIntN(args,kOfxImageEffectPropRenderWindow,4,window));
    check(props->propGetInt(image.handle,kOfxImagePropRowBytes,0,&stride));check(props->propGetPointer(image.handle,kOfxImagePropData,0,&data));
    auto magnitude=stride<0?-std::int64_t(stride):std::int64_t(stride);
    if(!data||bounds[0]!=0||bounds[1]!=0||bounds[2]!=static_cast<int>(frame.width)||bounds[3]!=static_cast<int>(frame.height)||
       magnitude<std::int64_t(frame.width)*16||std::uint64_t(magnitude)*(frame.height-1)+frame.width*16>std::uint64_t(PTRDIFF_MAX)||
       window[0]<0||window[1]<0||window[2]<window[0]||window[3]<window[1]||window[2]>bounds[2]||window[3]>bounds[3])throw std::runtime_error("Unsupported output bounds/stride/render window");
    for(int y=window[1];y<window[3];++y){
        if(effects->abort(h))return kOfxStatFailed;
        auto* row=static_cast<std::uint8_t*>(data)+std::ptrdiff_t(y)*stride;
        for(int x=window[0];x<window[2];++x){
            const auto pixel=std::size_t(frame.height-1-y)*frame.width+static_cast<unsigned>(x);
            const float a=frame.value(pixel,3);float rgba[4]={frame.value(pixel,0)*a,frame.value(pixel,1)*a,frame.value(pixel,2)*a,a};
            std::memcpy(row+std::ptrdiff_t(x)*16,rgba,16);
        }
    }
    return kOfxStatOK;
}
OfxStatus entry(const char* action,const void* raw,OfxPropertySetHandle in,OfxPropertySetHandle out){
    auto h=reinterpret_cast<OfxImageEffectHandle>(const_cast<void*>(raw));
    try{
        if(!action)return kOfxStatErrValue;
        if(std::strcmp(action,kOfxActionLoad)==0){
            if(!host)return kOfxStatErrMissingHostFeature;
            effects=static_cast<OfxImageEffectSuiteV1*>(const_cast<void*>(host->fetchSuite(host->host,kOfxImageEffectSuite,1)));
            props=static_cast<OfxPropertySuiteV1*>(const_cast<void*>(host->fetchSuite(host->host,kOfxPropertySuite,1)));
            params=static_cast<OfxParameterSuiteV1*>(const_cast<void*>(host->fetchSuite(host->host,kOfxParameterSuite,1)));
            messages=static_cast<OfxMessageSuiteV1*>(const_cast<void*>(host->fetchSuite(host->host,kOfxMessageSuite,1)));
            return effects&&props&&params?kOfxStatOK:kOfxStatErrMissingHostFeature;
        }
        if(std::strcmp(action,kOfxActionUnload)==0)return kOfxStatOK;
        if(!effects||!props||!params)return kOfxStatErrMissingHostFeature;
        if(std::strcmp(action,kOfxActionDescribe)==0){describe(h);return kOfxStatOK;}
        if(std::strcmp(action,kOfxImageEffectActionDescribeInContext)==0){context(h);return kOfxStatOK;}
        if(std::strcmp(action,kOfxActionCreateInstance)==0){create(h);return kOfxStatOK;}
        if(std::strcmp(action,kOfxActionDestroyInstance)==0){auto* p=instance(h);check(props->propSetPointer(properties(h),kOfxPropInstanceData,0,nullptr));delete p;return kOfxStatOK;}
        if(std::strcmp(action,kOfxImageEffectActionGetClipPreferences)==0){text(out,kOfxImageEffectPropPreMultiplication,kOfxImagePreMultiplied);return kOfxStatOK;}
        if(std::strcmp(action,kOfxImageEffectActionGetFramesNeeded)==0)return framesNeeded(h,in,out);
        if(std::strcmp(action,kOfxImageEffectActionRender)==0)return render(h,in);
        return kOfxStatReplyDefault;
    }catch(const std::bad_alloc&){return kOfxStatErrMemory;}
    catch(const std::exception& e){if(messages)messages->message(h,kOfxMessageError,"CorridorKeyCache","%s",e.what());return kOfxStatFailed;}
    catch(...){return kOfxStatFailed;}
}
void setHost(OfxHost* value){host=value;}
OfxPlugin plugin={kOfxImageEffectPluginApi,1,"com.mikehall.corridorkey.cached",0,1,setHost,entry};
}
extern "C" {
OfxExport int OfxGetNumberOfPlugins(){return 1;}
OfxExport OfxPlugin* OfxGetPlugin(int index){return index==0?&plugin:nullptr;}
}
