#include "cache.hpp"
#include <iostream>
int main(int argc,char** argv) {
    try {
        if(argc!=3)throw std::runtime_error("cache_probe FILE INDEX");
        auto f=ck::frame(std::filesystem::u8path(argv[1]),static_cast<std::uint32_t>(std::stoul(argv[2])));
        std::cout<<f.width<<" "<<f.height<<" "<<f.count<<" "<<f.sourceStart<<" "<<f.fps;
        for(int c=0;c<4;++c)std::cout<<" "<<f.value(0,c);
        std::cout<<"\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
