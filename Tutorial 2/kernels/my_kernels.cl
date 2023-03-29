// Histogram using the scatter pattern
kernel void simpleHist(global const uchar* A, global int* B) {
	int id = get_global_id(0);

	// Assumes that H has been initialised to 0
	int binIndex = A[id];
	// Deals with race conditions
	atomic_inc(&B[binIndex]);
}


// LUT using the map pattern
kernel void LUT(global int* A, global int* B) {
	int id = get_global_id(0);
	B[id] = A[id] * (double)255 / A[255];
}

// Copies all pixels from A to B
kernel void ReProject(global uchar* A, global int* LUT, global uchar* B) {
	int id = get_global_id(0);
	B[id] = LUT[A[id]];
}

// Cumulative histogram using the hillis-steele scan algorithm
kernel void scan_hs(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	global int* C;

	for (int stride = 1; stride < N; stride *= 2) {
		B[id] = A[id];
		if (id >= stride)
			B[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

		C = A; A = B; B = C; //swap A & B between steps
	}
}