// int please();

typedef struct RTNNState {
    unsigned int numQueries = 0;
} RTNNState;

RTNNState* generate_state();
// void 

void please();

int testing();

float** getNeighborList(float* a, int numPoints, float radius, int timestep);
double calculate_radius(double p1x, double p1y, double p1z, double p2x, double p2y, double p2z);