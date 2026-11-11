// Ground truth for the base-side mesh deformation port: runs Kimera-PGMO's
// deformation::deformPoints (the call Hydra's backend makes) on raw arrays.
//   pgmo_deform <dir> <k> <tol_t>
// reads <dir>/{ctrl_pos.f64 (N,3), ctrl_stamp.u64 (N), ctrl_val.f64 (N,8: valid,x,y,z,qx,qy,qz,qw),
//              vert_pos.f32 (M,3), vert_stamp.u64 (M), [indices.u64]} and writes <dir>/out_pos.f32
#include <fstream>
#include <iostream>
#include <vector>
#include <set>
#include <gtsam/geometry/Pose3.h>
#include <kimera_pgmo/mesh_types.h>

// ADL hooks live in the cloud's namespace (global)
struct StampedCloud {
  std::vector<Eigen::Vector3f> pos;
  std::vector<uint64_t> stamps;
};
using kimera_pgmo::traits::Pos;
using kimera_pgmo::traits::VertexTraits;
size_t pgmoNumVertices(const StampedCloud& c) { return c.pos.size(); }
kimera_pgmo::traits::VertexProperties pgmoGetVertexProperties(const StampedCloud&) { return {}; }
Pos pgmoGetVertex(const StampedCloud& c, size_t i, VertexTraits* = nullptr) { return c.pos[i]; }
void pgmoSetVertex(StampedCloud& c, size_t i, const Pos& p, const VertexTraits* = nullptr) { c.pos[i] = p; }
uint64_t pgmoGetVertexStamp(const StampedCloud& c, size_t i) { return c.stamps[i]; }

#include <kimera_pgmo/mesh_deformation.h>

template <typename T>
std::vector<T> readAll(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return {};
  const auto n = f.tellg();
  f.seekg(0);
  std::vector<T> out(n / sizeof(T));
  f.read(reinterpret_cast<char*>(out.data()), n);
  return out;
}

int main(int argc, char** argv) {
  const std::string d = argv[1];
  const size_t k = std::stoul(argv[2]);
  const double tol = std::stod(argv[3]);
  const char prefix = argc > 4 ? argv[4][0] : 's';
  auto cp = readAll<double>(d + "/ctrl_pos.f64");
  auto cs = readAll<uint64_t>(d + "/ctrl_stamp.u64");
  auto cv = readAll<double>(d + "/ctrl_val.f64");
  auto vp = readAll<float>(d + "/vert_pos.f32");
  auto vs = readAll<uint64_t>(d + "/vert_stamp.u64");
  auto idx = readAll<uint64_t>(d + "/indices.u64");
  const size_t N = cs.size(), M = vs.size();
  std::vector<gtsam::Point3> ctrl(N);
  gtsam::Values values;
  for (size_t i = 0; i < N; ++i) {
    ctrl[i] = gtsam::Point3(cp[3 * i], cp[3 * i + 1], cp[3 * i + 2]);
    const double* v = &cv[8 * i];
    if (v[0] > 0.5) {
      values.insert(gtsam::Symbol(prefix, i),
                    gtsam::Pose3(gtsam::Rot3(v[7], v[4], v[5], v[6]), gtsam::Point3(v[1], v[2], v[3])));
    }
  }
  StampedCloud in, out;
  in.pos.resize(M);
  in.stamps = vs;
  for (size_t i = 0; i < M; ++i) in.pos[i] = Eigen::Vector3f(vp[3 * i], vp[3 * i + 1], vp[3 * i + 2]);
  out = in;
  std::vector<size_t> indices(idx.begin(), idx.end());
  std::vector<std::set<size_t>> cmap;
  kimera_pgmo::deformation::deformPoints(out, cmap, in, prefix, ctrl, std::vector<kimera_pgmo::Timestamp>(cs.begin(), cs.end()),
                                         values, k, tol, indices.empty() ? nullptr : &indices);
  std::ofstream f(d + "/out_pos.f32", std::ios::binary);
  for (size_t i = 0; i < M; ++i) f.write(reinterpret_cast<const char*>(out.pos[i].data()), 3 * sizeof(float));
  std::cerr << "deformed " << (indices.empty() ? M : indices.size()) << " of " << M << " vertices with " << values.size() << " control values\n";
  return 0;
}
