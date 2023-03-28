// Histogram using the scatter pattern
kernel void simpleHist(global const uchar* A, global int* H) {
	int id = get_global_id(0);

	// Assumes that H has been initialised to 0
	int binIndex = A[id];
	// Deals with race conditions
	atomic_inc(&H[binIndex]);
}

// Cumulative histogram using the scan pattern
kernel void cumulativeHist(global int* H, global int* CH) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id + 1; i < N; i++)
		atomic_add(&CH[i], H[id]);
}

// LUT using the map pattern
kernel void LUT(global int* CH, global int* LUT) {
	int id = get_global_id(0);
	LUT[id] = CH[id] * (double)255 / CH[255];
}

// Copies all pixels from A to B
kernel void ReProject(global uchar* A, global int* LUT, global uchar* B) {
	int id = get_global_id(0);
	B[id] = LUT[A[id]];
}