// Independent full-path fixed-design checks for double-branch and joint-base cuts.
// This probe never calls the production separator or modifies the tested model's rules.
#include "mlnd/formulation.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
namespace {
using namespace mlnd;
void require(bool x,const std::string& text){if(!x)throw std::runtime_error(text);}
template<class T> Matrix<T> filled(int r,int c,T x){Matrix<T> a(r,c);std::fill(a.values.begin(),a.values.end(),x);return a;}
Input fixture(int bases,const std::string& mode,bool shared_delivery=false){
 Input d;d.n_l=bases;d.n_h=3;d.n_od=2;d.n_levels=1;d.max_orders=2;
 d.name="branch_joint_validity";d.model_version="consecutive";d.route_mode="direct";d.flow_policy="aggregate";
 d.demand={4,7};d.ground_cost={5,5};d.deadline={.18,.18};d.base_cost.assign(bases,.5);d.slot_epsilon.assign(3,.05);
 int N=d.n_nodes();d.flight_times=filled(N,N,.04);d.allowed_links=filled(N,N,1);d.link_cost=filled(N,N,0.);
 for(int i=0;i<N;++i){d.flight_times(i,i)=0;d.allowed_links(i,i)=0;for(int j=0;j<N;++j)if(i!=j&&(i<bases||j<bases))d.flight_times(i,j)=.03;}
 d.rider_mean=filled(2,3,1.);d.rider_mean(0,0)=d.rider_mean(1,0)=.02;
 d.first_cost=filled(2,3,.3);d.terminal_cost=filled(2,3,.1);
 d.receipt_mean=filled(2,3,1.);d.receipt_mean(0,1)=.01;d.receipt_mean(1,shared_delivery?1:2)=.01;
 d.removal_mean=filled(2,3,.01);d.slot_counts=filled(3,1,20);d.slot_cost=filled(3,1,.1);d.occupancy_limit=filled(3,1,14.072024748341311);
 d.kappa=filled(2,2,0.);if(mode=="AB"){d.kappa(0,1)=.8;d.kappa(1,0)=.4;}else{d.kappa(0,0)=.5;d.kappa(1,1)=.7;}
 d.delta=.04;d.operating_limit=1;d.fleet_cost=1;d.flight_cost=.2;d.waiting_cost=.3;return d;
}
struct Certificate {int r=0,s=-1,l=0,b=0;std::vector<std::pair<int,int>> blocked;};
bool absence(const Input& d,const PathCollection& paths,const Certificate& c){
 (void)d;
 for(const auto&p:paths.service_paths){
  if(p.b[0]!=c.l||p.b[1]!=c.b||p.first_od!=c.r||p.second_od!=c.s)continue;
  bool hit=false;for(auto[u,v,count]:p.arc_counts)if(count>0&&std::find(c.blocked.begin(),c.blocked.end(),std::make_pair(u,v))!=c.blocked.end())hit=true;
  if(!hit)return false;
 }
 return true;
}
std::vector<double> design(const Input&d,int a0,int a1,const std::vector<Certificate>& certs,int closedmask){
 DesignLayout z(d);int N=d.n_nodes(),L=d.n_l;std::vector<double>x(z.count,0);
 for(int i=0;i<N;++i)x[i]=1;for(int i=0;i<d.n_h;++i)x[z.capacity+i*d.n_levels]=1;
 if(a0>=0)x[z.assignment+a0]=1;if(a1>=0)x[z.assignment+L*L+a1]=1;
 for(int u=0;u<N;++u)for(int v=0;v<N;++v)x[z.airline+u*N+v]=d.allowed_links(u,v);
 for(std::size_t j=0;j<certs.size();++j)if(closedmask&(1<<j))for(auto[u,v]:certs[j].blocked)x[z.airline+u*N+v]=0;
 return x;
}
double rhs(const Input&d,const std::vector<double>&x,const std::vector<Certificate>&certs){
 DesignLayout z(d);int L=d.n_l,N=d.n_nodes();std::set<int> origins,triggers;std::set<std::pair<int,int>> block;
 for(const auto&c:certs){origins.insert(c.l);triggers.insert(z.assignment+(c.r*L+c.l)*L+c.b);if(c.s>=0)triggers.insert(z.assignment+(c.s*L+c.l)*L+c.b);block.insert(c.blocked.begin(),c.blocked.end());}
 double baseline=std::inner_product(d.demand.begin(),d.demand.end(),d.ground_cost.begin(),0.0),bound=baseline;
 for(int r=0;r<d.n_od;++r)for(int l=0;l<L;++l)if(!origins.count(l))for(int b=0;b<L;++b)bound-=d.demand[r]*d.ground_cost[r]*x[z.assignment+(r*L+l)*L+b];
 double gate=0;for(int a:triggers)gate+=1-x[a];for(auto[u,v]:block)gate+=x[z.airline+u*N+v];
 return bound-baseline*gate;
}
Json compare(const Input&d,const PathCollection&paths,const std::vector<double>&x,const std::vector<Certificate>&certs,const std::string&label,bool conditional){
 require(paths.complete&&paths.support_complete,"Full catalogue required");require(point_residuals(design_model(d).matrix,x).at("passed"),"Invalid strategic design");
 std::set<int>origins;for(const auto&c:certs){require(absence(d,paths,c),"Blocking set misses original complete branch");require(d.demand[c.r]>0,"Zero trigger demand");if(c.s>=0)require(d.effective_delta()>0&&d.demand[c.s]>0&&d.kappa(c.r,c.s)>0,"Nonpositive double branch");origins.insert(c.l);}
 require(origins.size()==certs.size(),"Independent probe expects at most one certificate per origin");
 auto f=build_formulation(d,paths,&x,nullptr,nullptr,false,conditional,true);SolverOptions options;options.threads=1;options.seconds=30;options.gap=1e-9;
 NativeSolver old_solver(f.matrix,options,"branch_joint_original");auto a=old_solver.optimize(30);
 require(a.has_solution&&a.optimal,label+": original optimum not certified: "+a.status);require(repair_inventory_primal(f,a.values).at("passed"),label+": old primal repair");auto residual=point_residuals(f.matrix,a.values);require(residual.at("passed"),label+": original residual");
 double old_cost=objective_value(f.matrix,a.values),joint=rhs(d,x,certs),tol=2e-7*(1+std::abs(old_cost));
 Json singles=Json::array();std::vector<double>bounds;
 for(const auto&c:certs){double v=rhs(d,x,{c});singles.push_back(v);bounds.push_back(v);}bounds.push_back(joint);
 for(std::size_t j=0;j<bounds.size();++j){require(old_cost+tol>=bounds[j],label+": cut excludes original optimum");Coefficients row;for(int i=0;i<static_cast<int>(f.matrix.variables.size());++i)if(f.matrix.variables[i].cost!=0)row.emplace_back(i,f.matrix.variables[i].cost);f.matrix.row("independent_branch_joint_"+std::to_string(j),'G',bounds[j],std::move(row));}
 NativeSolver strengthened(f.matrix,options,"branch_joint_added");auto b=strengthened.optimize(30);require(b.has_solution&&b.optimal,label+": strengthened optimum not certified: "+b.status);require(repair_inventory_primal(f,b.values).at("passed"),label+": new repair");auto newres=point_residuals(f.matrix,b.values);require(newres.at("passed"),label+": new residual");double new_cost=objective_value(f.matrix,b.values);require(std::abs(old_cost-new_cost)<=tol,label+": objective changed");
 double flow=0;for(int v:f.flow)if(v>=0)flow+=a.values[v];Json anchors=Json::array();for(const auto&c:certs)anchors.push_back({{"r",c.r},{"s",c.s},{"l",c.l},{"b",c.b},{"blocked",c.blocked}});
 Json out={{"case",label},{"conditional",conditional},{"service_paths",paths.service_paths.size()},{"complete_catalogue",true},{"certificates",anchors},{"individual_rhs",singles},{"joint_rhs",joint},{"original_objective",old_cost},{"strengthened_objective",new_cost},{"absolute_difference",std::abs(old_cost-new_cost)},{"objective_tolerance",tol},{"original_service_task_flow",flow},{"original_residual",residual},{"strengthened_residual",newres},{"original_rho",Json::array()}};for(int v:f.rho)out["original_rho"].push_back(a.values[v]);
 std::cout<<label<<" conditional="<<conditional<<" Q="<<old_cost<<" joint="<<joint<<" diff="<<std::abs(old_cost-new_cost)<<std::endl;return out;
}
std::vector<Certificate> saved_certs(const Input&d,const PathCollection&paths,const std::vector<double>&x){
 DesignLayout z(d);int L=d.n_l,N=d.n_nodes();std::vector<Certificate>out;
 for(int l=0;l<L;++l){bool found=false;for(int b=0;b<L&&!found;++b)for(int r=0;r<d.n_od&&!found;++r)if(x[z.assignment+(r*L+l)*L+b]>.5&&d.demand[r]>0)for(int s=0;s<d.n_od&&!found;++s)if(x[z.assignment+(s*L+l)*L+b]>.5&&d.demand[s]>0&&d.kappa(r,s)>0){
  Certificate c;c.r=r;c.s=s;c.l=l;c.b=b;for(int u=0;u<N;++u)for(int v=0;v<N;++v)if(u!=v&&d.allowed_links(u,v)&&x[z.airline+u*N+v]<.5)c.blocked.emplace_back(u,v);
  if(absence(d,paths,c)){out.push_back(c);found=true;}
 }}require(!out.empty(),"Saved bad candidate has no required blocked double branch");return out;
}
}
int main(int argc,char**argv){try{
 if(argc!=3)throw std::invalid_argument("Usage: branch_joint_probe CASES_JSON OUTPUT_JSON");Json checks=Json::array();
 for(const std::string mode:{"AB","AA"}){
  auto d=fixture(1,mode);auto paths=enumerate_paths(d);Certificate c;c.r=0;c.s=mode=="AB"?1:0;c.blocked={{d.n_l+1,d.n_l}};
  for(auto[a,b]:std::vector<std::pair<int,int>>{{0,0},{0,-1},{-1,0},{-1,-1}})for(int mask:{0,1})for(bool conditional:{false,true})checks.push_back(compare(d,paths,design(d,a,b,{c},mask),{c},"single_base_"+mode+"_a"+std::to_string(a)+"_b"+std::to_string(b)+"_mask"+std::to_string(mask),conditional));
 }
 for(const std::string mode:{"double","mixed","single","shared_arc"}){
  bool shared=mode=="shared_arc";auto d=fixture(2,"AA",shared);auto paths=enumerate_paths(d);Certificate c0,c1;c0.r=c0.s=0;c0.l=c0.b=0;c1.r=c1.s=1;c1.l=c1.b=1;c0.blocked={{d.n_l+1,d.n_l}};c1.blocked={{d.n_l+(shared?1:2),d.n_l}};
  if(mode=="mixed"||mode=="single"){c0.s=-1;c0.blocked={{0,d.n_l}};}if(mode=="single"){c1.s=-1;c1.blocked={{1,d.n_l}};}
  std::vector<Certificate>cs{c0,c1};for(auto[a,b]:std::vector<std::pair<int,int>>{{0,3},{0,-1},{1,3},{-1,3}})for(int mask:{0,1,2,3})for(bool conditional:{false,true})checks.push_back(compare(d,paths,design(d,a,b,cs,mask),cs,"two_bases_"+mode+"_a"+std::to_string(a)+"_b"+std::to_string(b)+"_mask"+std::to_string(mask),conditional));
 }
 Json cases;std::ifstream input(argv[1]);input>>cases;for(const auto&c:cases){auto d=read_input(c.at("input").get<std::string>(),2);auto paths=enumerate_paths(d);std::vector<double>bad;for(char b:c.at("bad_design_key").get<std::string>()){require(b=='0'||b=='1',"Design encoding");bad.push_back(b-'0');}auto cs=saved_certs(d,paths,bad);auto positive=solution_design(d,c.at("positive_solution"));for(bool conditional:{false,true}){checks.push_back(compare(d,paths,bad,cs,c.at("label").get<std::string>()+"_saved_bad",conditional));checks.push_back(compare(d,paths,positive,cs,c.at("label").get<std::string>()+"_saved_positive",conditional));}}
 Json output={{"passed",true},{"comparisons",checks},{"comparison_count",checks.size()},{"scope","Independent full original path domain; fixed designs before versus after adding double/joint cost rows; no operating model change"}};std::ofstream out(argv[2]);out<<output.dump(2)<<'\n';
 }catch(const IloException&e){std::cerr<<e<<'\n';return 1;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
