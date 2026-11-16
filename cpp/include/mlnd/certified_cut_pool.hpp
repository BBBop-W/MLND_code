#pragma once
#include "mlnd/linear_model.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace mlnd {
// Only unconditional, previously certified affine cuts belong here. Sleeping
// replaces the RHS by an outward-rounded bound implied by variable bounds;
// row identities remain stable for all other separators and solver mappings.
class CertifiedCutPool {
    using Key=std::pair<double,Coefficients>;
    struct Entry {
        RowRecord original;
        int row=0;
        std::size_t last_binding=0;
        double redundant_rhs=0;
        bool active=true, can_sleep=false;
    };
    std::set<Key> keys_;
    std::vector<Entry> entries_;
    std::size_t epoch_=0,duplicates_=0,suspensions_=0,reactivations_=0;
    static Key key(const RowRecord& row) {
        if(row.sense!='G' || !std::isfinite(row.rhs))
            throw std::invalid_argument("Cut pool requires a finite certified >= row");
        auto coefficients=row.coefficients;
        for(auto [j,a]:coefficients)
            if(j<0 || !std::isfinite(a))throw std::invalid_argument("Invalid certified cut coefficient");
        std::sort(coefficients.begin(),coefficients.end());
        return {row.rhs,std::move(coefficients)};
    }
    static double slack(const RowRecord& row,const std::vector<double>& point) {
        long double value=-static_cast<long double>(row.rhs);
        for(auto [j,a]:row.coefficients) {
            if(static_cast<std::size_t>(j)>=point.size() || !std::isfinite(point[j]))
                throw std::invalid_argument("Invalid cut-pool evaluation point");
            value+=static_cast<long double>(a)*point[j];
        }
        return static_cast<double>(value);
    }
    static double tolerance(const RowRecord& row,const std::vector<double>& point) {
        long double scale=1+std::abs(static_cast<long double>(row.rhs));
        for(auto [j,a]:row.coefficients)scale+=std::abs(static_cast<long double>(a)*point.at(j));
        return 1e-8*static_cast<double>(scale);
    }
public:
    bool duplicate(const RowRecord& row) {
        if(!keys_.count(key(row)))return false;
        ++duplicates_;return true;
    }
    void remember(const RowRecord& row,int index,const LinearModel& model) {
        keys_.insert(key(row));
        Entry entry;entry.original=row;entry.row=index;entry.last_binding=epoch_;
        long double bound=0;
        bool finite=true;
        for(auto [j,a]:row.coefficients) {
            const auto& v=model.variables.at(j);
            const double endpoint=a>=0?v.lower:v.upper;
            if(a==0)continue;
            if(!std::isfinite(endpoint) || std::abs(endpoint)>=IloInfinity) {finite=false;break;}
            const auto negative_infinity=-std::numeric_limits<long double>::infinity();
            const long double term=std::nextafter(static_cast<long double>(a)*endpoint,negative_infinity);
            bound=std::nextafter(bound+term,negative_infinity);
        }
        const double rounded=std::nextafter(static_cast<double>(bound),-INFINITY);
        entry.can_sleep=finite && std::isfinite(rounded) && rounded<=row.rhs;
        entry.redundant_rhs=rounded;
        entries_.push_back(std::move(entry));
    }
    template<class SetRhs>
    std::size_t reactivate(const std::vector<double>& point,LinearModel& model,SetRhs set_rhs) {
        std::size_t count=0;
        for(auto& entry:entries_)if(!entry.active &&
            slack(entry.original,point)<-tolerance(entry.original,point)) {
            set_rhs(entry.row,'G',entry.original.rhs);
            model.rows.at(entry.row).rhs=entry.original.rhs;
            entry.active=true;entry.last_binding=epoch_;++count;
        }
        reactivations_+=count;return count;
    }
    template<class SetRhs>
    std::size_t suspend(const std::vector<double>& point,std::size_t active_limit,
                        std::size_t age,LinearModel& model,SetRhs set_rhs) {
        ++epoch_;
        std::vector<std::pair<double,std::size_t>> eligible;
        std::size_t active=0;
        for(std::size_t i=0;i<entries_.size();++i) {
            auto& entry=entries_[i];if(!entry.active)continue;
            ++active;
            const double margin=slack(entry.original,point);
            if(margin<=tolerance(entry.original,point))entry.last_binding=epoch_;
            else if(entry.can_sleep && epoch_-entry.last_binding>=age)
                eligible.emplace_back(-margin,i);
        }
        std::sort(eligible.begin(),eligible.end());
        std::size_t count=0;
        for(auto [score,index]:eligible) {
            (void)score;if(active<=active_limit)break;
            auto& entry=entries_[index];
            set_rhs(entry.row,'G',entry.redundant_rhs);
            model.rows.at(entry.row).rhs=entry.redundant_rhs;
            entry.active=false;--active;++count;
        }
        suspensions_+=count;return count;
    }
    template<class SetRhs>
    void restore_all(LinearModel& model,SetRhs set_rhs) {
        for(auto& entry:entries_)if(!entry.active) {
            set_rhs(entry.row,'G',entry.original.rhs);
            model.rows.at(entry.row).rhs=entry.original.rhs;entry.active=true;
            ++reactivations_;
        }
    }
    Json metadata() const {
        std::size_t active=0;for(const auto& e:entries_)active+=e.active;
        return {{"registered",entries_.size()},{"active",active},{"dormant",entries_.size()-active},
            {"exact_duplicates_skipped",duplicates_},{"suspensions",suspensions_},{"reactivations",reactivations_},
            {"epochs",epoch_},{"physical_rows_removed",0},{"scope","unconditional_certified_affine_cuts"},
            {"sleep_method","redundant_rhs_with_stable_row_indices"}};
    }
};
} // namespace mlnd
