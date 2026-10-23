// CorridorKey native completed-frame reader. No inference or host state here.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ck {
using Digest = std::array<std::uint8_t,32>;
class Sha256 {
    std::array<std::uint32_t,8> h{{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}};
    std::array<std::uint8_t,64> block{};
    std::uint64_t length=0;
    std::size_t used=0;
    static std::uint32_t r(std::uint32_t x,int n) {return (x>>n)|(x<<(32-n));}
    void transform() {
        static constexpr std::uint32_t k[64]={
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        std::uint32_t w[64]{};
        for(int i=0;i<16;++i) for(int j=0;j<4;++j) w[i]=(w[i]<<8)|block[4*i+j];
        for(int i=16;i<64;++i) w[i]=w[i-16]+(r(w[i-15],7)^r(w[i-15],18)^(w[i-15]>>3))+w[i-7]+(r(w[i-2],17)^r(w[i-2],19)^(w[i-2]>>10));
        auto a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],v=h[7];
        for(int i=0;i<64;++i) {
            auto t=v+(r(e,6)^r(e,11)^r(e,25))+((e&f)^(~e&g))+k[i]+w[i];
            auto u=(r(a,2)^r(a,13)^r(a,22))+((a&b)^(a&c)^(b&c));
            v=g;g=f;f=e;e=d+t;d=c;c=b;b=a;a=t+u;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=v;
    }
public:
    void add(const std::uint8_t* p,std::size_t n) {
        length+=n;
        while(n) {auto take=std::min(n,64-used);std::memcpy(block.data()+used,p,take);used+=take;p+=take;n-=take;if(used==64){transform();used=0;}}
    }
    Digest finish() {
        auto bits=length*8;block[used++]=0x80;
        if(used>56){while(used<64)block[used++]=0;transform();used=0;}
        while(used<56)block[used++]=0;
        for(int i=7;i>=0;--i)block[used++]=static_cast<std::uint8_t>(bits>>(i*8));
        transform();Digest out{};
        for(int i=0;i<8;++i)for(int j=0;j<4;++j)out[i*4+j]=static_cast<std::uint8_t>(h[i]>>(24-j*8));
        return out;
    }
};
inline std::uint64_t le(const std::uint8_t* p,int n) {std::uint64_t v=0;for(int i=n-1;i>=0;--i)v=(v<<8)|p[i];return v;}
inline float scalar(const std::uint8_t* p) {auto u=static_cast<std::uint32_t>(le(p,4));float v;std::memcpy(&v,&u,4);return v;}
struct Frame {
    std::uint32_t width=0,height=0,count=0;
    std::uint64_t sourceStart=0;
    double fps=0;
    bool bound=false;
    std::vector<Digest> sourceDigests;
    std::vector<std::uint8_t> pixels;
    float value(std::size_t pixel,int channel) const {return scalar(pixels.data()+pixel*16+channel*4);}
};
inline void read(std::ifstream& in,std::uint8_t* p,std::size_t n) {
    if(!in.read(reinterpret_cast<char*>(p),static_cast<std::streamsize>(n)))throw std::runtime_error("Incomplete cache");
}
inline Frame frame(const std::filesystem::path& path,std::uint32_t index) {
    static_assert(sizeof(float)==4 && sizeof(double)==8,"IEEE float sizes required");
    static_assert(std::numeric_limits<float>::is_iec559,"IEEE floats required");
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("Cache unavailable");
    std::array<std::uint8_t,100> h{};read(in,h.data(),h.size());
    const bool bound=std::memcmp(h.data(),"CKOFX002",8)==0;
    if(!bound&&std::memcmp(h.data(),"CKOFX001",8)!=0)throw std::runtime_error("Unsupported cache");
    Frame f;f.bound=bound;f.width=static_cast<std::uint32_t>(le(h.data()+8,4));f.height=static_cast<std::uint32_t>(le(h.data()+12,4));
    f.count=static_cast<std::uint32_t>(le(h.data()+16,4));f.sourceStart=le(h.data()+20,8);
    auto bits=le(h.data()+28,8);std::memcpy(&f.fps,&bits,8);
    const auto pixels=std::uint64_t(f.width)*f.height;
    if(!f.width||!f.height||!f.count||f.count>(bound?16384u:1000000u)||index>=f.count||pixels>16*1024*1024||f.sourceStart>std::uint64_t(INT64_MAX)||!std::isfinite(f.fps)||f.fps<=0)throw std::runtime_error("Invalid cache bounds");
    if(bound&&f.sourceStart>std::uint64_t(INT64_MAX)-(f.count-1))throw std::runtime_error("Bound source range exceeds integer frame limits");
    const auto bytes=pixels*16;
    const auto entryBytes=bound?64u:32u;
    const auto tableBytes=std::size_t(f.count)*entryBytes;
    const auto size=100+tableBytes+bytes*f.count;
    in.seekg(0,std::ios::end);if(in.tellg()!=static_cast<std::streamoff>(size))throw std::runtime_error("Cache size mismatch");
    in.seekg(100);std::vector<std::uint8_t> table(tableBytes);read(in,table.data(),table.size());
    Sha256 header;header.add(h.data(),68);header.add(table.data(),table.size());auto hd=header.finish();
    if(std::memcmp(hd.data(),h.data()+68,32))throw std::runtime_error("Cache header checksum mismatch");
    in.seekg(static_cast<std::streamoff>(100+tableBytes+bytes*index));f.pixels.resize(static_cast<std::size_t>(bytes));read(in,f.pixels.data(),f.pixels.size());
    std::uint8_t at[4];for(int i=0;i<4;++i)at[i]=static_cast<std::uint8_t>(index>>(8*i));
    Sha256 hash;hash.add(h.data()+36,32);hash.add(at,4);hash.add(f.pixels.data(),f.pixels.size());auto digest=hash.finish();
    if(std::memcmp(digest.data(),table.data()+index*entryBytes,32))throw std::runtime_error("Cache frame checksum mismatch");
    if(bound){f.sourceDigests.resize(f.count);for(std::size_t i=0;i<f.count;++i)std::memcpy(f.sourceDigests[i].data(),table.data()+i*64+32,32);}
    for(std::size_t i=0;i<f.pixels.size();i+=4){const auto v=scalar(f.pixels.data()+i);if(!std::isfinite(v)||(i%16==12&&(v<0||v>1)))throw std::runtime_error("Invalid cache pixel");}
    return f;
}
} // namespace ck
