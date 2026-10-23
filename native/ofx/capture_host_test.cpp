// Actual adapter actions through official suites. This is not a Resolve host.
#define main legacy_cache_fixture_main
#include "host_test.cpp"
#undef main
#include "ofxProgress.h"
#include <locale>
#if !defined(_WIN32)
#include <csignal>
#include <sys/resource.h>
#endif

namespace capture_mock {
using namespace mock;
std::string scenario;
int starts=0,ends=0,updates=0,sets=0;
std::filesystem::path captureParent;
struct Comma:std::numpunct<char>{char do_decimal_point()const override{return ',';}};
OfxStatus currentValue(OfxParamHandle h,...){auto& param=*reinterpret_cast<Param*>(h);va_list ap;va_start(ap,h);if(param.kind==kOfxParamTypeString)*va_arg(ap,char**)=const_cast<char*>(param.text.c_str());else *va_arg(ap,int*)=param.number;va_end(ap);return kOfxStatOK;}
OfxStatus setValue(OfxParamHandle h,...){++sets;if(scenario=="setter-failure")return kOfxStatFailed;auto& param=*reinterpret_cast<Param*>(h);va_list ap;va_start(ap,h);param.text=va_arg(ap,const char*);va_end(ap);return kOfxStatOK;}
OfxStatus begin(void*,const char*){++starts;return kOfxStatOK;}
OfxStatus begin2(void* effect,const char* label,const char*){return begin(effect,label);}
OfxStatus update(void*,double fraction){++updates;require(fraction>=0&&fraction<=1,"bad progress fraction");return scenario=="cancel"&&fraction>.2?kOfxStatReplyNo:kOfxStatOK;}
OfxStatus end(void* raw){++ends;
    if(scenario=="final-abort")reinterpret_cast<Effect*>(raw)->abort=true;
    if(scenario=="manifest-conflict")for(const auto& entry:std::filesystem::directory_iterator(captureParent))if(entry.is_directory()){std::ofstream file(entry.path()/"capture.json");file<<"UNKNOWN FILE MUST SURVIVE";}
    return scenario=="end-failure"?kOfxStatFailed:kOfxStatOK;}
OfxStatus captureImage(OfxImageClipHandle clip,OfxTime time,const OfxRectD* region,OfxPropertySetHandle* out){require(region&&region->x1==0&&region->y1==0&&region->x2==5&&region->y2==3,"capture must request complete Source RoD explicitly");return image(clip,time,region,out);}
OfxProgressSuiteV1 progress1{begin,update,end};OfxProgressSuiteV2 progress2{begin2,update,end};
const void* suites(OfxPropertySetHandle h,const char* name,int version){if(std::strcmp(name,kOfxProgressSuite)==0){if(scenario=="missing-progress")return nullptr;if(version==2&&scenario!="v1")return &progress2;if(version==1)return &progress1;}return fetch(h,name,version);}
}
int main(int argc,char** argv){
    using namespace capture_mock;
    try{
        require(argc==3,"capture_host_test PARENT SCENARIO");scenario=argv[2];captureParent=std::filesystem::u8path(argv[1]);
        if(scenario=="locale")std::locale::global(std::locale(std::locale::classic(),new Comma));
#if !defined(_WIN32)
        if(scenario=="write-failure"){struct rlimit limit{};require(::getrlimit(RLIMIT_FSIZE,&limit)==0,"getrlimit");limit.rlim_cur=128;require(::setrlimit(RLIMIT_FSIZE,&limit)==0,"setrlimit");std::signal(SIGXFSZ,SIG_IGN);}
#endif
        setup();effectSuite.clipGetImage=captureImage;paramSuite.paramGetValue=currentValue;paramSuite.paramSetValue=setValue;
        Properties hostProps;hostProps.ints[kOfxImageEffectPropTemporalClipAccess]={scenario=="no-temporal"?0:1};OfxHost host{handle(hostProps),suites};auto* plugin=OfxGetPlugin(0);plugin->setHost(&host);require(plugin->mainEntry(kOfxActionLoad,nullptr,nullptr,nullptr)==kOfxStatOK,"load");
        Effect effect;auto h=reinterpret_cast<OfxImageEffectHandle>(&effect);
        auto call=[&](const char* action,Properties* args=nullptr,Properties* out=nullptr){return plugin->mainEntry(action,h,args?handle(*args):nullptr,out?handle(*out):nullptr);};
        require(call(kOfxActionDescribe)==kOfxStatOK,"describe");require(call(kOfxImageEffectActionDescribeInContext)==kOfxStatOK,"context");require(call(kOfxActionCreateInstance)==kOfxStatOK,"create");
        effect.params["CacheFile"].text="PREVIOUS-CACHE-UNCHANGED";effect.params["CacheStart"].number=37;
        effect.params["CaptureParent"].text=argv[1];effect.params["CaptureIn"].number=-2;effect.params["CaptureOutExclusive"].number=1;effect.params["CaptureMaxMiB"].number=512;effect.params["CaptureLinear709Declared"].number=scenario=="undeclared"?0:1;
        if(scenario=="range")effect.params["CaptureOutExclusive"].number=200;
        if(scenario=="budget")effect.params["CaptureMaxMiB"].number=0;
        auto& source=effect.clips["Source"];source.props.ints[kOfxImageClipPropConnected]={1};source.props.doubles[kOfxImagePropPixelAspectRatio]={1};source.props.doubles[kOfxImageEffectPropFrameRate]={23.976};source.props.doubles[kOfxImageEffectPropFrameRange]={-2,0};source.props.doubles["width"]={5};source.props.doubles["height"]={3};
        if(scenario=="large-budget"){source.props.doubles["width"]={32768};source.props.doubles["height"]={512};}
        if(scenario=="fps")source.props.doubles[kOfxImageEffectPropFrameRate]={std::numeric_limits<double>::quiet_NaN()};
        constexpr unsigned width=5,height=3;const int magnitude=width*16+12;std::vector<std::vector<std::uint8_t>> buffers(3);
        for(int frame=0;frame<3;++frame){
            const int stride=frame==1?-magnitude:magnitude;auto& buffer=buffers[frame];buffer.resize(std::size_t(magnitude)*height,0x6b);auto* base=buffer.data()+(stride<0?std::size_t(magnitude)*(height-1):0);
            for(unsigned top=0;top<height;++top)for(unsigned x=0;x<width;++x){float rgba[4]={float(frame)+float(top)*.25f+float(x)*.125f,(top==0&&x==0)?-0.f:-2.f+float(top)*.5f,4.f+frame,1.f};if(frame==1&&top==0&&x==0&&scenario=="alpha")rgba[3]=.5f;if(frame==1&&top==0&&x==0&&scenario=="nan")rgba[0]=std::numeric_limits<float>::quiet_NaN();std::memcpy(base+std::ptrdiff_t(height-1-top)*stride+std::size_t(x)*16,rgba,16);}
            auto& image=source.images[double(frame-2)];image.strings[kOfxImageEffectPropPixelDepth]={kOfxBitDepthFloat};image.strings[kOfxImageEffectPropComponents]={kOfxImageComponentRGBA};image.strings[kOfxImagePropField]={scenario=="field"?kOfxImageFieldUpper:kOfxImageFieldNone};image.strings[kOfxImageEffectPropPreMultiplication]={kOfxImageOpaque};image.doubles[kOfxImagePropPixelAspectRatio]={scenario=="image-par"?2.:1.};image.doubles[kOfxImageEffectPropRenderScale]={scenario=="image-scale"?.5:1.,1};image.ints[kOfxImagePropBounds]={0,0,int(width),int(height)};image.ints[kOfxImagePropRowBytes]={scenario=="stride"?1:stride};image.pointers[kOfxImagePropData]={base};
        }
        Properties args;args.strings[kOfxPropType]={kOfxTypeParameter};args.strings[kOfxPropName]={scenario=="other-param"?"CacheFile":"CaptureUpstream"};args.strings[kOfxPropChangeReason]={scenario=="plugin-edited"?kOfxChangePluginEdited:kOfxChangeUserEdited};args.doubles[kOfxPropTime]={-2};args.doubles[kOfxImageEffectPropRenderScale]={scenario=="action-scale"?.5:1.,1};
        bool recursiveVisited=false;
        source.onRead=[&]{if(scenario=="recursive"&&!recursiveVisited){recursiveVisited=true;Properties nested;const auto value=call(kOfxImageEffectActionGetRegionOfDefinition,&args,&nested);require(value==kOfxStatReplyDefault||value==kOfxStatOK,"recursive RoD");}};
        if(scenario=="abort")effect.abort=true;
        auto result=call(kOfxActionInstanceChanged,&args);if(scenario=="repeat"&&result==kOfxStatOK)result=call(kOfxActionInstanceChanged,&args);
        require(acquired==released,"leaked capture host image");require(effect.params["CacheFile"].text=="PREVIOUS-CACHE-UNCHANGED"&&effect.params["CacheStart"].number==37,"previous cache changed");
        const bool last=!effect.params["LastCaptureManifest"].text.empty();require(call(kOfxActionDestroyInstance)==kOfxStatOK,"destroy");require(plugin->mainEntry(kOfxActionUnload,nullptr,nullptr,nullptr)==kOfxStatOK,"unload");
        std::cout<<"{\"status\":"<<result<<",\"acquired\":"<<acquired<<",\"released\":"<<released<<",\"starts\":"<<starts<<",\"ends\":"<<ends<<",\"updates\":"<<updates<<",\"sets\":"<<sets<<",\"last\":"<<(last?"true":"false")<<",\"recursive\":"<<(recursiveVisited?"true":"false")<<"}\n";
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
