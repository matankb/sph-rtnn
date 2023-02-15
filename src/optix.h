// int please();

typedef struct RTNNState {
    unsigned int numQueries = 0;
} RTNNState;

RTNNState* generate_state();
// void 

void please();

int testing();

float** getNeighborList(fptype* a, fptype* radii, int numPoints, float radius, double max_interactions, int timestep);
double calculate_radius(double p1x, double p1y, double p1z, double p2x, double p2y, double p2z);