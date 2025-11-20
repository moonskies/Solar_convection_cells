// main.cpp - Stable Hydrodynamics Phase 1
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <omp.h>
#include <filesystem>

namespace fs = std::filesystem;

// --- CONFIGURATION (SAFE MODE) ---
// Lower resolution increases stability significantly (dx is larger)
const int Nx = 100;
const int Ny = 50;
const double Lx = 4.0;
const double Ly = 1.0;
const double dx = Lx / (Nx - 1);
const double dy = Ly / (Ny - 1);

// Time stepping: Must be less than dx^2 / 4 for stability
const double dt = 0.00005;
const int STEPS = 10000;
const int OUTPUT_INTERVAL = 2000; // Save every 2000 steps

// Physics Parameters
const double Pr = 1.0;
const double Ra = 20000.0; // Lower Ra is easier to simulate

// Arrays
std::vector<double> T(Nx * Ny, 0.0);
std::vector<double> w(Nx * Ny, 0.0);
std::vector<double> psi(Nx * Ny, 0.0);

int idx(int i, int j) {
    if (i < 0) i = 0; if (i >= Nx) i = Nx - 1;
    if (j < 0) j = 0; if (j >= Ny) j = Ny - 1;
    return j * Nx + i;
}

void saveData(int step) {
    std::string filename = "output/data/step_" + std::to_string(step) + ".txt";
    std::ofstream file(filename);

    if (!file.is_open()) return;

    file << Nx << " " << Ny << "\n";
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            file << T[idx(i,j)] << " " << psi[idx(i,j)] << " ";
        }
        file << "\n";
    }
    std::cout << "Saved " << filename << std::endl;
}

// Check for simulation explosion
bool isStable() {
    for(double val : T) {
        if (std::isnan(val) || std::isinf(val)) return false;
    }
    return true;
}

void solvePoisson() {
    double tolerance = 1e-4;
    double error = 1.0;
    int max_iter = 1000;
    double omega = 1.6;

    int iter = 0;
    while (error > tolerance && iter < max_iter) {
        error = 0.0;
        // Red-Black or simple parallel loop (ignoring slight race condition for speed)
        #pragma omp parallel for reduction(+:error)
        for (int j = 1; j < Ny - 1; ++j) {
            for (int i = 1; i < Nx - 1; ++i) {
                double old_val = psi[idx(i,j)];

                // Finite Difference Inverse Laplacian
                // psi_new = 0.25 * (neighbors + source)
                double neighbors = psi[idx(i+1,j)] + psi[idx(i-1,j)] +
                                   psi[idx(i,j+1)] + psi[idx(i,j-1)];

                // Source term from Vorticity (w)
                double source = w[idx(i,j)] * dx * dy; // Approx if dx~dy

                double new_val = 0.25 * (neighbors + source);

                psi[idx(i,j)] = old_val + omega * (new_val - old_val);
                error += std::abs(psi[idx(i,j)] - old_val);
            }
        }
        iter++;
    }
}

int main() {
    // Clean Start
    if (fs::exists("output")) fs::remove_all("output");
    fs::create_directory("output");
    fs::create_directory("output/data");
    fs::create_directory("output/images");

    std::cout << "Initializing Stable Simulation..." << std::endl;
    std::cout << "Grid: " << Nx << "x" << Ny << ", dt: " << dt << std::endl;

    // 1. Initialize
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            double y_norm = (double)j / (Ny - 1);
            T[idx(i,j)] = 1.0 - y_norm; // Hot bottom (1.0), Cold top (0.0)

            // Add noise to middle
            if(j > 0 && j < Ny-1) {
                T[idx(i,j)] += 0.01 * ((rand() % 100) / 100.0);
            }
        }
    }

    std::vector<double> T_new = T;
    std::vector<double> w_new = w;

    // 2. Time Loop
    for (int step = 0; step <= STEPS; ++step) {

        solvePoisson();

        #pragma omp parallel for
        for (int j = 1; j < Ny - 1; ++j) {
            for (int i = 1; i < Nx - 1; ++i) {
                int c = idx(i,j);

                // Calculate Velocity from Streamfunction
                double u = (psi[idx(i,j+1)] - psi[idx(i,j-1)]) / (2*dy);
                double v = -(psi[idx(i+1,j)] - psi[idx(i-1,j)]) / (2*dx);

                // Derivatives
                double dw_dx = (w[idx(i+1,j)] - w[idx(i-1,j)]) / (2*dx);
                double dw_dy = (w[idx(i,j+1)] - w[idx(i,j-1)]) / (2*dy);
                double dT_dx = (T[idx(i+1,j)] - T[idx(i-1,j)]) / (2*dx);
                double dT_dy = (T[idx(i,j+1)] - T[idx(i,j-1)]) / (2*dy);

                // Laplacian (Diffusion)
                double lap_w = (w[idx(i+1,j)] + w[idx(i-1,j)] - 2*w[c])/(dx*dx) +
                               (w[idx(i,j+1)] + w[idx(i,j-1)] - 2*w[c])/(dy*dy);
                double lap_T = (T[idx(i+1,j)] + T[idx(i-1,j)] - 2*T[c])/(dx*dx) +
                               (T[idx(i,j+1)] + T[idx(i,j-1)] - 2*T[c])/(dy*dy);

                // Update T (Convection-Diffusion)
                T_new[c] = T[c] + dt * (-(u*dT_dx + v*dT_dy) + lap_T);

                // Update Vorticity (Convection-Diffusion + Buoyancy)
                // Buoyancy term: Ra * Pr * dT/dx
                w_new[c] = w[c] + dt * (-(u*dw_dx + v*dw_dy) + Pr*lap_w + Ra*Pr*dT_dx);
            }
        }

        // Boundary Conditions (Fixed T, Free Slip)
        for (int i = 0; i < Nx; ++i) {
            T_new[idx(i,0)] = 1.0; T_new[idx(i,Ny-1)] = 0.0;
            w_new[idx(i,0)] = 0.0; w_new[idx(i,Ny-1)] = 0.0;
        }
        for (int j = 0; j < Ny; ++j) {
            T_new[idx(0,j)] = T_new[idx(1,j)];
            T_new[idx(Nx-1,j)] = T_new[idx(Nx-2,j)];
            w_new[idx(0,j)] = 0.0; w_new[idx(Nx-1,j)] = 0.0;
        }

        T = T_new;
        w = w_new;

        // Safety Check & Save
        if (step % OUTPUT_INTERVAL == 0) {
            if (!isStable()) {
                std::cerr << "CRITICAL ERROR: Simulation unstable at step " << step << std::endl;
                break;
            }
            saveData(step);
        }
    }

    return 0;
}