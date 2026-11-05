#pragma once
#include "mlnd/paths.hpp"
#include <type_traits>

namespace mlnd {
namespace identity_detail {
// Exact local snapshots, never persisted or interpreted as a portable format.
// Serialize fields individually: padding in structs/tuples is not an identity.
template<class T> std::enable_if_t<std::is_arithmetic_v<T>> append(std::string& out,const T& value) {
    out.append(reinterpret_cast<const char*>(&value),sizeof(value));
}
inline void append(std::string& out,const std::string& value) {
    append(out,value.size());out.append(value);
}
template<class A,class B> void append(std::string&,const std::pair<A,B>&);
template<class... T> void append(std::string&,const std::tuple<T...>&);
template<class T,std::size_t N> void append(std::string&,const std::array<T,N>&);
template<class T> void append(std::string&,const std::vector<T>&);
template<class A,class B> void append(std::string& out,const std::pair<A,B>& value) {
    append(out,value.first);append(out,value.second);
}
template<class... T> void append(std::string& out,const std::tuple<T...>& value) {
    std::apply([&](const auto&... fields){(append(out,fields),...);},value);
}
template<class T,std::size_t N> void append(std::string& out,const std::array<T,N>& value) {
    for(const auto& x:value)append(out,x);
}
template<class T> void append(std::string& out,const std::vector<T>& value) {
    append(out,value.size());for(const auto& x:value)append(out,x);
}
template<class... T> std::string snapshot(const T&... fields) {
    std::string out;(append(out,fields),...);return out;
}
}
inline std::string exact_path_identity(const ServicePath& p) {
    return identity_detail::snapshot(p.id,p.segment_id,p.b,p.first_od,p.second_od,p.mode,p.pickups,p.deliveries,
        p.legs,p.served_counts,p.arc_counts,p.flight_time,p.time,p.wait,p.slot_times,p.pickup_times,p.delivery_times,p.cost);
}
inline std::string exact_path_identity(const EmptyPath& p) {
    return identity_detail::snapshot(p.id,p.b,p.legs,p.arc_counts,p.flight_time,p.time,p.cost);
}
} // namespace mlnd
