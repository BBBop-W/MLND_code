#include "mlnd/input.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#ifdef __APPLE__
#include <CommonCrypto/CommonDigest.h>
#else
#include <openssl/sha.h>
#endif

namespace mlnd {
namespace {
constexpr const char* schema = "consecutive-csv/1";
using Row = std::vector<std::string>;
using Rows = std::vector<Row>;
using Index = std::unordered_map<std::string, int>;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error("Instance: " + message);
}
void require(bool condition, const std::string& message) {
    if (!condition) fail(message);
}
Json read_json(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) fail("cannot open " + path.string());
    try { return Json::parse(stream); }
    catch (const Json::exception& error) { fail(path.string() + ": " + error.what()); }
}

// RFC4180 field parsing, including quoted commas, quotes and newlines.
Rows read_csv(const std::filesystem::path& path, const Row& header) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) fail("cannot open " + path.string());
    Rows rows;
    Row row;
    std::string field;
    bool quoted = false, closed = false, touched = false;
    char c;
    auto field_end = [&]() { row.push_back(field); field.clear(); closed = false; touched = false; };
    auto row_end = [&]() { field_end(); rows.push_back(row); row.clear(); };
    while (stream.get(c)) {
        if (quoted) {
            if (c == '"') {
                if (stream.peek() == '"') { stream.get(c); field += '"'; }
                else { quoted = false; closed = true; }
            } else field += c;
        } else if (c == ',') field_end();
        else if (c == '\n') row_end();
        else if (c == '\r') {
            if (stream.peek() == '\n') stream.get(c);
            row_end();
        } else if (c == '"' && !touched && field.empty() && !closed) {
            quoted = true; touched = true;
        } else {
            require(!closed && c != '"', "malformed quoted CSV field: " + path.string());
            field += c; touched = true;
        }
    }
    require(!quoted, "unterminated CSV quote: " + path.string());
    if (!row.empty() || touched || closed || !field.empty()) row_end();
    require(!rows.empty() && rows.front() == header, "wrong CSV header: " + path.string());
    rows.erase(rows.begin());
    for (const auto& r : rows)
        require(r.size() == header.size(), "wrong CSV column count: " + path.string());
    return rows;
}

double number(const std::string& text, const std::string& field, bool allow_inf = false) {
    std::size_t consumed = 0;
    double value;
    try { value = std::stod(text, &consumed); }
    catch (const std::exception&) { fail("invalid number in " + field + ": " + text); }
    require(consumed == text.size() && !std::isnan(value) &&
            (std::isfinite(value) || (allow_inf && value > 0)), "invalid number in " + field);
    return value;
}
int integer(const std::string& text, const std::string& field) {
    std::size_t consumed = 0;
    long long value;
    try { value = std::stoll(text, &consumed); }
    catch (const std::exception&) { fail("invalid integer in " + field + ": " + text); }
    require(consumed == text.size() && value >= std::numeric_limits<int>::min() &&
            value <= std::numeric_limits<int>::max(), "invalid integer in " + field);
    return static_cast<int>(value);
}
Index index_of(const std::vector<std::string>& ids, const std::string& domain) {
    require(!ids.empty(), "empty ID domain: " + domain);
    Index index;
    for (int i = 0; i < static_cast<int>(ids.size()); ++i)
        require(!ids[i].empty() && index.emplace(ids[i], i).second, "duplicate or empty ID: " + domain);
    return index;
}
int lookup(const Index& ids, const std::string& id, const std::string& domain) {
    const auto found = ids.find(id);
    require(found != ids.end(), "unknown " + domain + " ID: " + id);
    return found->second;
}
Rows ordered_rows(Rows rows, const Index& index, const std::string& domain) {
    require(rows.size() == index.size(), "wrong row count: " + domain);
    Rows ordered(rows.size());
    std::vector<bool> seen(rows.size(), false);
    for (auto& row : rows) {
        const int i = lookup(index, row.at(0), domain);
        require(!seen[i], "duplicate row: " + domain);
        seen[i] = true; ordered[i] = std::move(row);
    }
    return ordered;
}
bool close(double a, double b, double atol = 1e-14, double rtol = 1e-14) {
    return std::abs(a-b) <= atol + rtol * std::abs(b);
}
void nonnegative(const std::vector<double>& values, const std::string& field) {
    for (double v : values) require(std::isfinite(v) && v >= 0, field + " must be finite and nonnegative");
}
long double poisson_tail(int capacity, long double mean) {
    if (mean == 0) return 0;
    // A log-scaled CDF avoids underflow when the supplied capacity is large.
    const int mode = std::min(capacity, static_cast<int>(std::floor(mean)));
    const long double log_mode = -mean + mode * std::log(mean) - std::lgamma(static_cast<long double>(mode + 1));
    long double sum = 1, term = 1;
    for (int k = mode; k > 0; --k) { term *= k / mean; sum += term; }
    term = 1;
    for (int k = mode + 1; k <= capacity; ++k) { term *= mean / k; sum += term; }
    return std::max(0.0L, -std::expm1(log_mode + std::log(sum)));
}
double poisson_threshold(int capacity, double epsilon) {
    long double lo = 0, hi = std::max(1.0L, static_cast<long double>(capacity) + 1);
    while (poisson_tail(capacity, hi) < epsilon) hi *= 2;
    for (int i = 0; i < 100; ++i) {
        const long double mid = (lo + hi) / 2;
        if (poisson_tail(capacity, mid) < epsilon) lo = mid; else hi = mid;
    }
    return static_cast<double>((lo+hi)/2);
}
const Json units = {
    {"coordinates", "km, local projected coordinates"}, {"demand", "orders/hour"},
    {"time_coefficients", "hour"},
    {"base_cost,slot_cost,link_cost", "CNY/hour per selected facility, level or link"},
    {"fleet_cost", "CNY/hour per drone"},
    {"first_cost,terminal_cost,ground_cost,last_rider_cost", "CNY/order"},
    {"flight_cost,waiting_cost", "CNY/drone-hour of flight or waiting"},
    {"slot_counts", "compartments"},
    {"occupancy_limit", "mean parcels present; not orders/hour"},
    {"slot_epsilon,pickup_share,kappa", "dimensionless"}
};
const Json capacity_definition = {
    {"version", "poisson-strict-overflow/1"},
    {"formula", "max mu >= 0 such that P(Poisson(mu) > slot_counts) <= slot_epsilon"},
    {"threshold_column", "coefficients/locker_levels.csv:occupancy_limit"},
    {"verification_atol", 1e-12}, {"verification_rtol", 1e-12}
};
const std::set<std::string> input_files = {
    "sites.csv", "od_nodes.csv", "od_demand.csv", "periods.csv", "metadata.json", "model.json",
    "coefficients/base_costs.csv", "coefficients/locker_levels.csv", "coefficients/links.csv",
    "coefficients/od_costs.csv", "coefficients/od_hub.csv", "coefficients/kappa.csv"
};
std::set<std::string> keys(const Json& object) {
    require(object.is_object(), "expected JSON object");
    std::set<std::string> result;
    for (auto it = object.begin(); it != object.end(); ++it) result.insert(it.key());
    return result;
}
} // namespace

std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) fail("cannot hash " + path.string());
#ifdef __APPLE__
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
#else
    SHA256_CTX context;
    SHA256_Init(&context);
#endif
    std::array<char, 65536> buffer{};
    while (stream) {
        stream.read(buffer.data(), buffer.size());
#ifdef __APPLE__
        CC_SHA256_Update(&context, buffer.data(), static_cast<CC_LONG>(stream.gcount()));
#else
        SHA256_Update(&context, buffer.data(), static_cast<std::size_t>(stream.gcount()));
#endif
    }
    std::array<unsigned char, 32> digest{};
#ifdef __APPLE__
    CC_SHA256_Final(digest.data(), &context);
#else
    SHA256_Final(digest.data(), &context);
#endif
    std::ostringstream hex;
    for (unsigned char c : digest) hex << std::hex << std::setw(2) << std::setfill('0') << int(c);
    return hex.str();
}

Input read_input(const std::filesystem::path& path, int max_orders, bool verify_hashes) {
    Input d;
    d.instance_path = std::filesystem::absolute(path);
    d.manifest = read_json(path / "manifest.json");
    require(d.manifest.at("schema") == schema && keys(d.manifest.at("files")) == input_files,
            "unsupported or incomplete manifest");
    for (const auto& file : input_files)
        if (verify_hashes) require(sha256_file(path/file) == d.manifest.at("files").at(file).get<std::string>(),
                                   "SHA-256 mismatch: " + file);
    d.model_json = read_json(path / "model.json");
    const auto& model = d.model_json;
    require(model.at("schema") == schema && model.at("units") == units, "unsupported schema or units");
    require(model.at("capacity_definition") == capacity_definition, "unsupported capacity_definition");
    const auto& ids = model.at("ids");
    require(keys(ids) == std::set<std::string>{"bases","hubs","origins","destinations","ods","levels","flight_nodes"},
            "unsupported ID domains");
    d.base_ids = ids.at("bases").get<Row>(); d.hub_ids = ids.at("hubs").get<Row>();
    d.od_ids = ids.at("ods").get<Row>(); d.origin_ids = ids.at("origins").get<Row>();
    d.destination_ids = ids.at("destinations").get<Row>(); d.level_ids = ids.at("levels").get<Row>();
    d.flight_node_ids = ids.at("flight_nodes").get<Row>();
    std::map<std::string, Index> indices;
    for (auto it = ids.begin(); it != ids.end(); ++it) indices[it.key()] = index_of(it.value().get<Row>(), it.key());
    Row flight_ids = d.base_ids;
    flight_ids.insert(flight_ids.end(), d.hub_ids.begin(), d.hub_ids.end());
    require(d.flight_node_ids == flight_ids, "flight nodes must be bases followed by hubs");
    Row node_ids = d.origin_ids;
    node_ids.insert(node_ids.end(), d.destination_ids.begin(), d.destination_ids.end());
    const auto node_index = index_of(node_ids, "OD nodes");
    const auto& values = model.at("values");
    require(keys(values) == std::set<std::string>{"n_l","n_h","n_od","delta","operating_limit","fleet_cost",
        "flight_cost","waiting_cost","max_orders","name","candidate_routes","route_mode","flow_policy","model_version"},
        "missing or unknown model scalar");
    for (const char* key : {"n_l", "n_h", "n_od", "max_orders"})
        require(values.at(key).is_number_integer(), std::string("integer scalar required: ") + key);
    d.n_l = values.at("n_l"); d.n_h = values.at("n_h"); d.n_od = values.at("n_od");
    d.n_levels = static_cast<int>(d.level_ids.size());
    require(d.n_l == static_cast<int>(d.base_ids.size()) && d.n_h == static_cast<int>(d.hub_ids.size()) &&
            d.n_od == static_cast<int>(d.od_ids.size()), "dimension mismatch");
    d.delta = values.at("delta"); d.operating_limit = values.at("operating_limit");
    d.fleet_cost = values.at("fleet_cost"); d.flight_cost = values.at("flight_cost"); d.waiting_cost = values.at("waiting_cost");
    nonnegative({d.delta,d.operating_limit,d.fleet_cost,d.flight_cost,d.waiting_cost}, "scalar coefficients");
    require(d.operating_limit > 0 && d.fleet_cost > 0, "operating_limit and fleet_cost must be positive");
    d.max_orders = values.at("max_orders");
    require(d.max_orders == 1 || d.max_orders == 2, "max_orders must be 1 or 2");
    require(model.at("cases") == Json{{"K1", {{"max_orders",1}}}, {"K2", {{"max_orders",2}}}},
            "cases may change max_orders only");
    require(max_orders >= 0 && max_orders <= 2, "requested max_orders must be 0, 1 or 2");
    if (max_orders) d.max_orders = max_orders;
    d.name = values.at("name"); d.route_mode = values.at("route_mode");
    d.flow_policy = values.at("flow_policy"); d.model_version = values.at("model_version");
    require(d.model_version == "consecutive", "unsupported model_version");
    require(d.route_mode == "direct" || d.route_mode == "simple", "unsupported route_mode");
    require(d.flow_policy == "aggregate" || d.flow_policy == "prefix", "unsupported flow_policy");
    d.has_candidate_routes = !values.at("candidate_routes").is_null();
    if (d.has_candidate_routes) d.candidate_routes = values.at("candidate_routes").get<std::vector<std::vector<int>>>();

    const auto sites = ordered_rows(read_csv(path/"sites.csv", {"site_id","type","x_km","y_km","activity_rate"}),
                                    indices.at("flight_nodes"), "sites");
    const auto nodes = ordered_rows(read_csv(path/"od_nodes.csv", {"node_id","role","x_km","y_km","raw_endpoint_count"}),
                                    node_index, "OD nodes");
    auto coordinates = [&](const Rows& rows, int offset, int size, const std::string& role) {
        Matrix<double> matrix(size,2);
        for (int i = 0; i < size; ++i) {
            const auto& row = rows.at(offset+i);
            require(row[1] == role, "wrong node role: " + role);
            matrix(i,0) = number(row[2], "x_km"); matrix(i,1) = number(row[3], "y_km");
        }
        return matrix;
    };
    d.bases = coordinates(sites,0,d.n_l,"L"); d.hubs = coordinates(sites,d.n_l,d.n_h,"H");
    d.origins = coordinates(nodes,0,d.origin_ids.size(),"origin");
    d.destinations = coordinates(nodes,d.origin_ids.size(),d.destination_ids.size(),"destination");
    const auto ods = ordered_rows(read_csv(path/"od_demand.csv", {"od_id","origin_node","destination_node","order_count",
        "exposure_hours","lambda_per_hour","direct_distance_km"}), indices.at("ods"), "OD demand");
    const auto periods = read_csv(path/"periods.csv", {"period_id","start_hour","end_hour","dates","exposure_hours",
        "order_count","total_rate_per_hour"});
    require(periods.size() == 1, "one pooled demand period is required");
    const auto& period = periods.front();
    const double start = number(period[1],"start_hour"), end = number(period[2],"end_hour");
    require(start >= 0 && start < end && end <= 24, "invalid period hour window");
    const double exposure = number(period[4],"exposure_hours");
    std::set<std::string> dates;
    std::istringstream date_stream(period[3]);
    std::string date;
    while (std::getline(date_stream,date,';')) require(!date.empty() && dates.insert(date).second, "empty or duplicate period date");
    require(exposure > 0 && exposure == dates.size() * (end-start), "period exposure mismatch");
    long long count_sum = 0;
    std::set<std::pair<int,int>> pairs;
    for (const auto& row : ods) {
        const int origin = lookup(indices.at("origins"),row[1],"origin");
        const int destination = lookup(indices.at("destinations"),row[2],"destination");
        require(pairs.emplace(origin,destination).second,"duplicate OD pair");
        const int count = integer(row[3],"order_count");
        const double rate = number(row[5],"lambda_per_hour");
        require(count >= 0 && number(row[4],"exposure_hours") == exposure && close(rate,count/exposure) &&
                number(row[6],"direct_distance_km") >= 0, "inconsistent OD count/exposure/distance");
        count_sum += count; d.demand.push_back(rate); d.od_origin.push_back(origin); d.od_destination.push_back(destination);
    }
    require(count_sum == integer(period[5],"order_count") &&
            close(number(period[6],"total_rate_per_hour"),count_sum/exposure),"period count/rate mismatch");

    const int n = d.n_nodes(), r = d.n_od, h = d.n_h, s = d.n_levels;
    d.base_cost.resize(d.n_l); d.ground_cost.resize(r); d.deadline.resize(r); d.slot_epsilon.assign(h,-1);
    d.slot_counts = Matrix<int>(h,s); d.slot_cost = Matrix<double>(h,s); d.occupancy_limit = Matrix<double>(h,s);
    d.allowed_links = Matrix<int>(n,n); d.flight_times = Matrix<double>(n,n); d.link_cost = Matrix<double>(n,n);
    d.rider_mean = Matrix<double>(r,h); d.first_cost = Matrix<double>(r,h); d.terminal_cost = Matrix<double>(r,h);
    d.receipt_mean = Matrix<double>(r,h); d.removal_mean = Matrix<double>(r,h); d.pickup_share = Matrix<double>(r,h);
    d.last_rider_cost = Matrix<double>(r,h); d.last_rider_mean = Matrix<double>(r,h); d.kappa = Matrix<double>(r,r);
    auto table = [&](const std::string& name, const Row& id_columns, const Row& domains, const Row& columns,
                     const auto& assign) {
        require(model.at("table_layouts").at(name) == Json{{"id_columns",id_columns},{"domains",domains},{"value_columns",columns}},
                "wrong table layout: " + name);
        Row header = id_columns; header.insert(header.end(),columns.begin(),columns.end());
        const auto rows = read_csv(path/("coefficients/"+name+".csv"),header);
        const int cols = domains.size() == 2 ? indices.at(domains[1]).size() : 1;
        const std::size_t expected = indices.at(domains[0]).size() * cols;
        require(rows.size() == expected,"wrong coefficient row count: " + name);
        std::vector<bool> seen(expected,false);
        for (const auto& row : rows) {
            const int i = lookup(indices.at(domains[0]),row[0],domains[0]);
            const int j = domains.size() == 2 ? lookup(indices.at(domains[1]),row[1],domains[1]) : 0;
            const int flat = i * cols + j;
            require(!seen[flat],"duplicate coefficient row: " + name); seen[flat] = true;
            assign(i,j,row,static_cast<int>(domains.size()));
        }
    };
    table("base_costs",{"base_id"},{"bases"},{"base_cost"},[&](int i,int,const Row& row,int c) {
        d.base_cost[i] = number(row[c],"base_cost");
    });
    table("locker_levels",{"hub_id","level_id"},{"hubs","levels"},
        {"slot_counts","slot_cost","slot_epsilon","occupancy_limit"},[&](int i,int j,const Row& row,int c) {
        d.slot_counts(i,j) = integer(row[c],"slot_counts"); d.slot_cost(i,j) = number(row[c+1],"slot_cost");
        const double epsilon = number(row[c+2],"slot_epsilon");
        require(d.slot_epsilon[i] < 0 || d.slot_epsilon[i] == epsilon,"inconsistent epsilon across levels");
        d.slot_epsilon[i] = epsilon; d.occupancy_limit(i,j) = number(row[c+3],"occupancy_limit");
    });
    table("links",{"from_site","to_site"},{"flight_nodes","flight_nodes"},
        {"allowed_links","flight_times","link_cost"},[&](int i,int j,const Row& row,int c) {
        require(row[c] == "0" || row[c] == "1","allowed_links must be 0/1");
        d.allowed_links(i,j) = row[c] == "1"; d.flight_times(i,j) = number(row[c+1],"flight_times",true);
        d.link_cost(i,j) = number(row[c+2],"link_cost");
    });
    table("od_costs",{"od_id"},{"ods"},{"ground_cost","deadline"},[&](int i,int,const Row& row,int c) {
        d.ground_cost[i] = number(row[c],"ground_cost"); d.deadline[i] = number(row[c+1],"deadline");
    });
    table("od_hub",{"od_id","hub_id"},{"ods","hubs"},
        {"rider_mean","first_cost","terminal_cost","receipt_mean","removal_mean","pickup_share","last_rider_cost","last_rider_mean"},
        [&](int i,int j,const Row& row,int c) {
        std::array<Matrix<double>*,8> matrices = {&d.rider_mean,&d.first_cost,&d.terminal_cost,&d.receipt_mean,
            &d.removal_mean,&d.pickup_share,&d.last_rider_cost,&d.last_rider_mean};
        for (std::size_t k = 0; k < matrices.size(); ++k) (*matrices[k])(i,j) = number(row[c+k],"od_hub coefficient");
    });
    table("kappa",{"first_od","second_od"},{"ods","ods"},{"kappa"},[&](int i,int j,const Row& row,int c) {
        d.kappa(i,j) = number(row[c],"kappa");
    });

    Json arrays = Json::object();
    auto array_spec = [&](const std::string& name, const std::vector<int>& shape, const std::string& dtype) {
        arrays[name] = {{"shape",shape},{"dtype",dtype}};
    };
    for (const char* name : {"demand","ground_cost","deadline"}) array_spec(name,{r},"float64");
    array_spec("base_cost",{d.n_l},"float64"); array_spec("slot_epsilon",{h},"float64");
    for (const char* name : {"flight_times","link_cost"}) array_spec(name,{n,n},"float64");
    array_spec("allowed_links",{n,n},"bool"); array_spec("kappa",{r,r},"float64");
    array_spec("slot_counts",{h,s},"int64"); array_spec("slot_cost",{h,s},"float64");
    for (const char* name : {"rider_mean","first_cost","terminal_cost","receipt_mean","removal_mean","pickup_share","last_rider_cost","last_rider_mean"})
        array_spec(name,{r,h},"float64");
    array_spec("bases",{d.n_l,2},"float64"); array_spec("hubs",{h,2},"float64");
    array_spec("origins",{static_cast<int>(d.origin_ids.size()),2},"float64");
    array_spec("destinations",{static_cast<int>(d.destination_ids.size()),2},"float64");
    array_spec("od_origin",{r},"int64"); array_spec("od_destination",{r},"int64");
    require(model.at("arrays") == arrays,"array shape or dtype mismatch");
    require(keys(model.at("table_layouts")) == std::set<std::string>{"base_costs","locker_levels","links","od_costs","od_hub","kappa"},
            "unexpected coefficient table");
    for (const auto* vector : {&d.demand,&d.ground_cost,&d.deadline,&d.base_cost,&d.slot_epsilon}) nonnegative(*vector,"vector coefficient");
    for (const auto* matrix : {&d.rider_mean,&d.first_cost,&d.terminal_cost,&d.receipt_mean,&d.removal_mean,&d.slot_cost,
        &d.link_cost,&d.kappa,&d.occupancy_limit,&d.pickup_share,&d.last_rider_cost,&d.last_rider_mean})
        nonnegative(matrix->values,"matrix coefficient");
    std::map<std::pair<int,double>,double> thresholds;
    for (int i = 0; i < h; ++i) {
        require(d.slot_epsilon[i] > 0 && d.slot_epsilon[i] < 1,"epsilon must lie strictly between 0 and 1");
        for (int j = 0; j < s; ++j) {
            require(d.slot_counts(i,j) >= 0,"negative compartment count");
            const auto key = std::make_pair(d.slot_counts(i,j),d.slot_epsilon[i]);
            auto found = thresholds.find(key);
            if (found == thresholds.end()) found = thresholds.emplace(key,poisson_threshold(key.first,key.second)).first;
            require(close(d.occupancy_limit(i,j),found->second,1e-12,1e-12),"Poisson occupancy threshold mismatch");
        }
    }
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
        require(d.flight_times(i,j) >= 0,"negative flight time");
        require(!d.allowed_links(i,j) || std::isfinite(d.flight_times(i,j)),"allowed link has infinite flight time");
        if (i == j) require(!d.allowed_links(i,j) && d.flight_times(i,j) == 0,"self links must be disabled with zero flight time");
    }
    for (double value : d.kappa.values) require(value <= 1,"kappa must be in [0,1]");
    for (std::size_t i = 0; i < d.rider_mean.values.size(); ++i) {
        require(d.removal_mean.values[i] <= d.receipt_mean.values[i] + 1e-12,"meal removal cannot follow final receipt");
        require(d.pickup_share.values[i] <= 1,"pickup_share must be in [0,1]");
    }
    for (const auto& route : d.candidate_routes) {
        require(!route.empty(),"empty candidate route");
        std::set<int> seen;
        for (int node : route) require(node >= 0 && node < n && seen.insert(node).second,"invalid candidate route node");
        for (std::size_t i = 1; i < route.size(); ++i)
            require(d.allowed_links(route[i-1],route[i]),"candidate route uses an unavailable link");
    }
    d.metadata = read_json(path/"metadata.json");
    require(d.metadata.at("exchange_schema") == schema && d.metadata.at("coordinate_system") == "local_km",
            "unsupported metadata or coordinates");
    return d;
}
} // namespace mlnd
