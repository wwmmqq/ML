#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <ctime>

void inline debug(int k) {
    char hhh;
    std::cin >> hhh;
    std::cout << "****" << k <<std::endl;
}

class BPNN {
private:
	const static int Layer1 = 784;
	const static int Layer2 = 100;
	const static int Layer3 = 10;

	double alpha = 0.35;
	int input[Layer1];
	int target[Layer3];

	double W1[Layer2][Layer1];
	double W2[Layer3][Layer2];
	double a1[Layer2];
	double a2[Layer3];
	double delta1[Layer2];
	double delta2[Layer3];
	double b1[Layer2];
	double b2[Layer3];

public:
	BPNN() {
		init();
	}
	BPNN(const BPNN&) = delete;
	BPNN& operator =(const BPNN&) = delete;
	~BPNN(){}

	void init()
	{
		srand((int)time(0) + rand());
		//rand()%1000 * 0.001 - 0.5;

		//W1
		for (int i = 0; i < Layer2; i++) {
			for (int j = 0; j < Layer1; j++) {
				W1[i][j] = rand()%1000 * 0.001 - 0.5;
			}
		}

		//b1
		for (int i = 0; i < Layer2; i++) {
			 b1[i] = rand()%1000 * 0.001 - 0.5;
		}

		//W2
		for (int i = 0; i < Layer3; i++) {
			for (int j = 0; j < Layer2; j++) {
				W2[i][j] = rand()%1000 * 0.001 - 0.5;
			}
		}

		//b2
		for (int i = 0; i < Layer3; i++) {
			 b2[i] = rand()%1000 * 0.001 - 0.5;
		}
	}

	double inline f(double x)
	{
		return 1.0 / (1.0 + exp(-x));
	}

	void f1()
	{
		for (int i = 0; i < Layer2; i++) {
			double z1 = 0.0;
			for (int j = 0; j < Layer1; j++) {
				z1 += W1[i][j] * input[j];
			}
			z1 += b1[i];
			a1[i] = f(z1);
		}
	}

	void f2()
	{
		for (int i = 0; i < Layer3; i++) {
			double z2 = 0.0;
			for (int j = 0; j < Layer2; j++) {
				z2 += W2[i][j] * a1[j];
			}
			z2 += b2[i];
			a2[i] = f(z2);
		}
	}

	void dt2() 
	{
		for (int i = 0; i < Layer3; i++) {
			delta2[i] = a2[i] * (1.0 - a2[i]) * (a2[i] - target[i]);
		}
	}

	void dt1()
	{
		for (int i = 0; i < Layer2; i++) {
			double temp = 0.0;
			for (int j = 0; j < Layer3; j++) {
				temp += W2[j][i] * delta2[j];	//按权重返回上一层的 delta
			}
			delta1[i] = a1[i] * (1.0 - a1[i]) * temp;
		}
	}

	void feedback_update_w2_b2()
	{
		for (int i = 0; i < Layer3; i++) {
			for (int j = 0; j < Layer2; j++) {
				W2[i][j] -= alpha * delta2[i] * a1[j];
			}
			b2[i] -= alpha * delta2[i] * b2[i]; 
		}
	}

	void feedback_update_w1_b1()
	{
		for (int i = 0; i < Layer2; i++) {
			for (int j = 0; j < Layer1; j++) {
				W1[i][j] -= alpha * delta1[i] * input[j]; 
			}
			b1[i] -= alpha * delta1[i] * b1[i];
		}
	}

	void train()
	{
		FILE *image_train;
		FILE *image_label;
		image_train = fopen("../data/mnist/train-images-idx3-ubyte", "rb");
		image_label = fopen("../data/mnist/train-labels-idx1-ubyte", "rb");
		if (image_train == NULL || image_label == NULL){
			std::cout << "can't open the file!" << std::endl;
			exit(0);
		}

		unsigned char image_buf[784];
		unsigned char label_buf[10];
		
		int useless[1000];
		fread(useless, 1, 16, image_train);
		fread(useless, 1, 8, image_label);

		std::cout << "training ...\n";
		//60000 datas
		int data_id = 0;
		while (!feof(image_train) && !feof(image_label)) {
			memset(image_buf, 0, 784);
			memset(label_buf, 0, 10);
			fread(image_buf, 1, 784, image_train);
			fread(label_buf, 1, 1, image_label);

			//initialize the input by 28 x 28 (0,1)matrix of the images
			for (int i = 0; i < 784; i++){
				if ((unsigned int)image_buf[i] < 128){
					input[i] = 0;
				}
				else{
					input[i] = 1;
				}
			}

			//initialize the target output
			int target_value = (unsigned int)label_buf[0];
			for (int k = 0; k < Layer3; k++){
				target[k] = 0;
			}
			target[target_value] = 1;

			//get the output and start training
			//std::cout << "image: "<< data_id <<"\n";
			
			f1();
			//debug(1);
			f2();
			//debug(2);
			dt2();
			//debug(3);
			dt1();
			//debug(4);
			feedback_update_w2_b2();
			//debug(5);
			feedback_update_w1_b1();
			//debug(6);

			data_id++;
			if (data_id % 1000 == 0){
				std::cout << "training image: " << data_id << std::endl;
			}
		}
	}

	void test()
	{
		std::cout << "testing ...\n";
		int test_success_count = 0;
		int test_num = 0;

		FILE *image_test;
		FILE *image_test_label;
		image_test = fopen("../data/mnist/t10k-images-idx3-ubyte", "rb");
		image_test_label = fopen("../data/mnist/t10k-labels-idx1-ubyte", "rb");
		if (image_test == NULL || image_test_label == NULL){
			std::cout << "can't open the file!" << std::endl;
			exit(0);
		}

		unsigned char image_buf[784];
		unsigned char label_buf[10];
		
		int useless[1000];
		fread(useless, 1, 16, image_test);
		fread(useless, 1, 8, image_test_label);

		while (!feof(image_test) && !feof(image_test_label)){
			memset(image_buf, 0, 784);
			memset(label_buf, 0, 10);
			fread(image_buf, 1, 784, image_test);
			fread(label_buf, 1, 1, image_test_label);

			//initialize the input by 28 x 28 (0,1)matrix of the images
			for (int i = 0; i < 784; i++){
				if ((unsigned int)image_buf[i] < 128){
					input[i] = 0;
				}
				else{
					input[i] = 1;
				}
			}

			//initialize the target output
			for (int k = 0; k < Layer3; k++){
				target[k] = 0;
			}
			int target_value = (unsigned int)label_buf[0];
			target[target_value] = 1;
			
			//get the ouput and compare with the targe
			f1();
			f2();

			double max_value = -99999;
			int max_index = 0;
			for (int k = 0; k < Layer3; k++){
				if (a2[k] > max_value){
					max_value = a2[k];
					max_index = k;
				}
			}

			//output == target
			if (target[max_index] == 1){
				test_success_count++;
			}
			
			test_num++;

			if ((int)test_num % 1000 == 0){
				std::cout << "test num: " << test_num << "  success: " << test_success_count << std::endl;
			}
		}
		std::cout << std::endl;
		std::cout << "The success rate: " << (test_success_count*1.0) / (test_num*1.0) << std::endl;
	}
};

int main(int argc, char const *argv[])
{
	BPNN bpnn;

	std::cout << "end: bpnn init\n";

	for(int epoch = 1; epoch < 10; epoch++) {
		std::cout << "******* epoch:  "<< epoch << std::endl;
		bpnn.train();
		bpnn.test();
	}

	return 0;
}
