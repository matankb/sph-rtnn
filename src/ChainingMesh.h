#ifndef TRIFORCE_CHAININGMESH_H
#define TRIFORCE_CHAININGMESH_H
/*
 * ChainingMesh.h
 *
 *
 *
 * Created On: 08/02/2022
 *
 * Last Updated:
 *    * AJK - 08/02/2022 - Initially created
 *    * RLM - 08/09/2022 - Filled in ChainingMesh for rectangular population
 */

#include "Utilities.h"
#include "Parameters.h"
#include "ParticleManagerBase.h"
#include "LinkedList.h"

constexpr int32_t NULL_LINK = -1;


struct Mesh {
  vec3<uint32_t> nn;   // number of physical nodes
  vec3<uint32_t> nc;   // number of physical cells
  vec3<uint32_t> nn_g; // number of nodes with ghost layers
  vec3<uint32_t> nc_g; // number of cells with ghost layers
  uint32_t nn_yz, nc_yz, nn_yz_g, nc_yz_g;
  uint32_t nGhost, nPerCell;
  uint32_t nNodes, nCells;
  vec3<uint32_t> start;
  vec3<uint32_t> stop;
  vec3<fptype> dx;

  fpvec_array node_loc;
  fpvec_array cell_loc;

  Mesh(uint32_t nx, uint32_t ny, uint32_t nz, uint32_t nGhost, uint32_t nPerCell) :
    nn(nx, ny, nz),
    nc(nn[0] - 1, nn[1] - 1, nn[2] - 1),
    nn_g(nn[0] + 2 * nGhost, nn[1] + 2 * nGhost, nn[2] + 2 * nGhost),
    nc_g(nn_g[0] - 1, nn_g[1] - 1, nn_g[2] - 1),
    nn_yz(nn[1] * nn[2]),
    nc_yz(nc[1] * nc[2]),
    nn_yz_g(nn_g[1] * nn_g[2]),
    nc_yz_g(nc_g[1] * nc_g[2]),
    nGhost(nGhost), nPerCell(nPerCell),
    nNodes(nn_g[0] * nn_g[1] * nn_g[2]),
    nCells(nc_g[0] * nc_g[1] * nc_g[2]),
    start(nGhost, nGhost, nGhost),
    stop(nGhost+nn[0], nGhost+nn[1], nGhost+nn[2]),
    dx(0.0, 0.0, 0.0)
  {
    node_loc = make_unique<vec3<fptype>[]>(nNodes);
    cell_loc = make_unique<vec3<fptype>[]>(nCells);
  };

  KOKKOS_FUNCTION uint32_t getPhysicalNidFromCoord(const uint32_t i, const uint32_t j, const uint32_t k) const {
    return k + j * nn.z() + i * nn_yz;
  }

  KOKKOS_FUNCTION uint32_t getPhysicalCidFromCoord(const uint32_t i, const uint32_t j, const uint32_t k) const {
    return k + j * nc.z() + i * nc_yz;
  }

  KOKKOS_FUNCTION uint32_t getLogicalNidFromCoord(const uint32_t i, const uint32_t j, const uint32_t k) const {
    return k + j * nn_g.z() + i * nn_yz_g;
  }

  KOKKOS_FUNCTION uint32_t getLogicalCidFromCoord(const uint32_t i, const uint32_t j, const uint32_t k) const {
    return k + j * nc_g.z() + i * nc_yz_g;
  }

  KOKKOS_FUNCTION vec3<uint32_t> getPhysicalCoordFromNid(const uint32_t idx) const {
    return {idx / nn_yz, (idx / nn.z()) % nn.y(), idx % nn.z()};
  }

  KOKKOS_FUNCTION vec3<uint32_t> getPhysicalCoordFromCid(const uint32_t idx) const {
    return {idx / nc_yz, (idx / nc.z()) % nc.y(), idx % nc.z()};
  }

  KOKKOS_FUNCTION vec3<uint32_t> getLogicalCoordFromNid(const uint32_t idx) const {
    return {idx / nn_yz_g, (idx / nn_g.z()) % nn_g.y(), idx % nn_g.z()};
  }

  KOKKOS_FUNCTION vec3<uint32_t> getLogicalCoordFromCid(const uint32_t idx) const {
    return {idx / nc_yz_g, (idx / nc_g.z()) % nc_g.y(), idx % nc_g.z()};
  }
};

class ChainingMesh {
private:
  const Parameters* params;
  const ParticleManagerBase* pm;

  void populateUniformRectangularMesh(Domain domain) const;

public:
  ChainingMesh(const Parameters* params_input, const ParticleManagerBase* pm_input);
  ~ChainingMesh() = default;

  void buildMesh() const;
  void cleanLinkedList() const;
  void fillInChainingMesh() const;

public:
  Mesh* mesh;
  unique_ptr<uint32_t[]> cidVec;
  unique_ptr<LinkedList<uint32_t>[]> llArray;
};

ChainingMesh::ChainingMesh(const Parameters* params_input, const ParticleManagerBase* pm_input) :
  params(params_input), pm(pm_input)
{
  // Do we like the following change? Instead of assigning the node size
  // of the mesh, we specify an average particles per cell to divide the grid.

  // mesh = make_unique<Mesh>(params->mesh.dims[0], params->mesh.dims[1], params->mesh.dims[2],
  //         static_cast<uint32_t>(params->mesh.nGhosts), static_cast<uint32_t>(pow(2, DIM)));

  vec3<uint32_t> ngrid = {2,2,2};
  for (uint32_t d = 0; d < DIM; d++) {
      ngrid[d] = params->p1.num_particles[d] / params->mesh.nppc[d] + 1;
  }

  mesh = new Mesh(ngrid[0], ngrid[1], ngrid[2], params->mesh.nGhosts, static_cast<uint32_t>(pow(2, DIM)));

  buildMesh();

  llArray = make_unique<LinkedList<uint32_t>[]>(mesh->nCells);
  cidVec = make_unique<uint32_t[]>(pm->maxSize);
}

void ChainingMesh::buildMesh() const {
  // Determine range of particles at current time
  vec3<fptype> min_range{PHIGH,PHIGH,PHIGH};
  vec3<fptype> max_range{PLOW,PLOW,PLOW};
  vec3<fptype> range_diff;

  for (uint32_t pid = 0; pid < pm->pNum; pid++) {
    Particle tmp = pm->getParticle(pid);
    vec3<fptype> loc = tmp.loc;
    for (uint32_t d = 0; d < 3; d++) {
      min_range[d] = fmin(min_range[d],loc[d]);
      max_range[d] = fmax(max_range[d],loc[d]);
      range_diff[d] = max_range[d] - min_range[d];
    }
  }

  fptype margin = 0.1;
  Domain particle_domain;
  particle_domain.dims = params->mesh.dims;
  particle_domain.nGhosts = params->mesh.nGhosts;
  particle_domain.x0 = min_range[0] - 0.5 * margin * range_diff[0];
  particle_domain.y0 = min_range[1] - 0.5 * margin * range_diff[1];
  particle_domain.z0 = min_range[2] - 0.5 * margin * range_diff[2];
  particle_domain.x1 = max_range[0] + 0.5 * margin * range_diff[0];
  particle_domain.y1 = max_range[1] + 0.5 * margin * range_diff[1];
  particle_domain.z1 = max_range[2] + 0.5 * margin * range_diff[2];

  populateUniformRectangularMesh(particle_domain);
}

void ChainingMesh::fillInChainingMesh() const {
  cleanLinkedList();
  buildMesh();

  vec3<fptype> x0 = mesh->node_loc[mesh->getLogicalNidFromCoord(mesh->start[0], mesh->start[1], mesh->start[2])];

  for (uint32_t pid = 0; pid < pm->pNum; pid++) {
    Particle tmp = pm->getParticle(pid);
    vec3<fptype> loc = tmp.loc;

    uint32_t cid = mesh->getLogicalCidFromCoord(
                      mesh->start[0] + uint32_t((loc.x() - x0.x()) / mesh->dx[0]),
                      mesh->start[1] + uint32_t((loc.y() - x0.y()) / mesh->dx[1]),
                      mesh->start[2] + uint32_t((loc.z() - x0.z()) / mesh->dx[2])
                    );

    llArray[cid].addAtHead(pid);
    cidVec[pid] = cid;
  }
}

void ChainingMesh::cleanLinkedList() const {
  for (uint32_t cid = 0; cid < mesh->nCells; cid++) {
    llArray[cid].deleteAllNodes();
  }
}

void ChainingMesh::populateUniformRectangularMesh(const Domain domain) const {
  vec3<fptype> length{(domain.x1 - domain.x0), (domain.y1 - domain.y0), (domain.z1 - domain.z0)};
  vec3<fptype> dw{};

  for (uint32_t i = 0; i < 3; i++) {
    dw[i] = (mesh->nc[i] > 0) ? length[i] / static_cast<fptype>(mesh->nc[i]) : 0.5 * length[i];
  }

  mesh->dx = dw;

  for (uint32_t n=0; n < mesh->nNodes; n++) {
    mesh->node_loc[n] = {0.0, 0.0, 0.0};
  }

  // Fill in the physical nodes on the logical grid
  for (uint32_t i = mesh->nGhost; i < (mesh->nn.x() + mesh->nGhost); i++) {
    for (uint32_t j = mesh->nGhost; j < (mesh->nn.y() + mesh->nGhost); j++) {
      for (uint32_t k = mesh->nGhost; k < (mesh->nn.z() + mesh->nGhost); k++) {
        uint32_t nid = mesh->getLogicalNidFromCoord(i, j, k);

        mesh->node_loc[nid] = {domain.x0 + (i - mesh->nGhost) * dw[0],
                                domain.y0 + (j - mesh->nGhost) * dw[1],
                                domain.z0 + (k - mesh->nGhost) * dw[2]};
      }
    }
  }

  // nG = 2
  // *   *   *
  //
  // *   *   *
  //
  // *   *   *
  //    1,2 2,2
  //
  // *   *   *

  // x - direction
  for (int32_t i = mesh->nGhost - 1; i >= 0; i--) { // 1, 0
    for (uint32_t j = mesh->nGhost; j < mesh->nGhost + mesh->nn.y(); j++) { // 2 -> end
      for (uint32_t k = mesh->nGhost; k < mesh->nGhost + mesh->nn.z(); k++) { // 2 -> end
        uint32_t nid    = mesh->getLogicalNidFromCoord(i, j, k);
        uint32_t nid_I1 = mesh->getLogicalNidFromCoord(i+1, j, k);
        uint32_t nid_I2 = mesh->getLogicalNidFromCoord(i+2, j, k);

        mesh->node_loc[nid] = static_cast<fptype>(2.0) * mesh->node_loc[nid_I1] - mesh->node_loc[nid_I2];

        uint32_t ir = mesh->nn_g.x() - 1 - i;

        uint32_t nidr = mesh->getLogicalNidFromCoord(ir, j, k);
        nid_I1 = mesh->getLogicalNidFromCoord(ir-1, j, k);
        nid_I2 = mesh->getLogicalNidFromCoord(ir-2, j, k);

        mesh->node_loc[nidr] = static_cast<fptype>(2.0) * mesh->node_loc[nid_I1] - mesh->node_loc[nid_I2];
      } // end for(k)
    } // end for(j)
  } // end for(i)

  // y - direction
  for (uint32_t i = mesh->nGhost; i < mesh->nGhost + mesh->nn.x(); i++) { // 2 -> end
    for (int32_t j = mesh->nGhost - 1; j >= 0; j--) { // 1, 0
      for (uint32_t k = mesh->nGhost; k < mesh->nGhost + mesh->nn.z(); k++) { // 2 -> end
        uint32_t nid    = mesh->getLogicalNidFromCoord(i, j, k);
        uint32_t nid_I1 = mesh->getLogicalNidFromCoord(i, j + 1, k);
        uint32_t nid_I2 = mesh->getLogicalNidFromCoord(i, j + 2, k);

        float twoPointZero = 2.0; // added by matan

        mesh->node_loc[nid] = twoPointZero * mesh->node_loc[nid_I1] - mesh->node_loc[nid_I2];

        uint32_t jr = mesh->nn_g.y() - 1 - j;

        uint32_t nidr = mesh->getLogicalNidFromCoord(i, jr, k);
        nid_I1 = mesh->getLogicalNidFromCoord(i, jr-1, k);
        nid_I2 = mesh->getLogicalNidFromCoord(i, jr-2, k);

        // float twoPointZero = 2.0;

        mesh->node_loc[nidr] = (fptype) 2.0 * mesh->node_loc[nid_I1] - mesh->node_loc[nid_I2];
      } // end for(k)
    } // end for(j)
  } // end for(i)

  // y - direction
  for (uint32_t i = mesh->nGhost; i < mesh->nGhost + mesh->nn.x(); i++) { // 2 -> end
    for (uint32_t j = mesh->nGhost; j < mesh->nGhost + mesh->nn.y(); j++) { // 2 -> end
      for (int32_t k = mesh->nGhost - 1; k >= 0; k--) { // 1, 0
        uint32_t nid    = mesh->getLogicalNidFromCoord(i, j, k);
        uint32_t nid_I1 = mesh->getLogicalNidFromCoord(i, j, k + 1);
        uint32_t nid_I2 = mesh->getLogicalNidFromCoord(i, j, k + 2);

        mesh->node_loc[nid] = ((float) 2.0) * mesh->node_loc[nid_I1] - mesh->node_loc[nid_I2];

        uint32_t kr = mesh->nn_g.z() - 1 - k;

        uint32_t nidr = mesh->getLogicalNidFromCoord(i, j, kr);
        nid_I1 = mesh->getLogicalNidFromCoord(i, j, kr-1);
        nid_I2 = mesh->getLogicalNidFromCoord(i, j, kr-2);

        mesh->node_loc[nidr] = ((float) 2.0) * mesh->node_loc[nid_I1] - mesh->node_loc[nid_I2];
      } // end for(k)
    } // end for(j)
  } // end for(i)

  // Fill in the cell nodes on the logical grid
  for (uint32_t i = 0; i < mesh->nc_g.x(); i++) {
    for (uint32_t j = 0; j < mesh->nc_g.y(); j++) {
      for (uint32_t k = 0; k < mesh->nc_g.z(); k++) {
        vec3<fptype> accum = {0.0, 0.0, 0.0};
        for (uint32_t ii = 0; ii < 2; ii++) {
          for (uint32_t jj = 0; jj < 2; jj++) {
            for (uint32_t kk = 0; kk < 2; kk++) {
                accum += mesh->node_loc[mesh->getLogicalNidFromCoord(i+ii, j+jj, k+kk)];
            } // for(kk)
          } // for(jj)
        } // for(ii)
        uint32_t cid = mesh->getLogicalCidFromCoord(i, j, k);
        mesh->cell_loc[cid] = accum / static_cast<fptype>(powf(2.0, fpDIM));
      } // for(k)
    } // for(j)
  } // for(i)
} // end populateUniformRectangularMesh

#endif //TRIFORCE_CHAININGMESH_H
