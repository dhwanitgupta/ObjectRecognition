#include <stdio.h>
#include <iostream>
#include <dirent.h>
#include <iomanip>      // std::setprecision
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include<stdlib.h>
#include<string.h>
#include<fstream>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include"svmlight.h"
using namespace cv;
using namespace std;
static const Size trainingPadding = Size(0, 0);
static const Size winStride = Size(8, 8);
static string svmModelFile = "svmlightmodel.dat";
static string descriptorVectorFile = "descriptorvector.dat";
static string toLowerCase(const string& in) {
	string t;
	for (string::const_iterator i = in.begin(); i != in.end(); ++i) {
		t += tolower(*i);
	}
	return t;
}

static void storeCursor(void) {
	printf("\033[s");
}

static void resetCursor(void) {
	printf("\033[u");
}
static void saveDescriptorVectorToFile(vector<float>& descriptorVector, vector<unsigned int>& _vectorIndices, string fileName) {
	printf("Saving descriptor vector to file '%s'\n", fileName.c_str());
	string separator = " "; // Use blank as default separator between single features
	fstream File;
	float percent;
	File.open(fileName.c_str(), ios::out);
	if (File.good() && File.is_open()) {
		printf("Saving descriptor vector features:\t");
		storeCursor();
		for (int feature = 0; feature < descriptorVector.size(); ++feature) {
			if ((feature % 10 == 0) || (feature == (descriptorVector.size()-1)) ) {
				percent = ((1 + feature) * 100 / descriptorVector.size());
				printf("%4u (%3.0f%%)", feature, percent);
				fflush(stdout);
				resetCursor();
			}
			File << descriptorVector.at(feature) << separator;
		}
		printf("\n");
		File << endl;
		File.flush();
		File.close();
	}
}
int main(){
	initModule_nonfree();

	string featuresFile("Hogfeatures");
	//fstream File;
	//File.open(featuresFile.c_str(), ios::out);
	HOGDescriptor hog;
	string dir = "Caltech_11classes/001.ak47", filepath;
	string dir1 = "Caltech_11classes/033.cd";
	DIR *dp;
	struct dirent *dirp;
	struct stat filestat;

	dp = opendir( dir.c_str() );
	Mat img;
	int i;
	vector<float> featureVector;
	int img_no = 0;
	/*while (dirp = readdir( dp ))
	{
		filepath = dir + "/" + dirp->d_name;
		img = imread(filepath);
		if(!img.data)
			continue;
		resize(img,img,Size(128,128),0,0,INTER_LINEAR );
		vector<Point> locations;
		hog.compute(img, featureVector, winStride, trainingPadding, locations);
		File << "+1";
		for(i=0;i<featureVector.size();i++)
			File <<" "<< i +  1 << ":"<< featureVector[i];
		File << endl;

		if(img_no > 30)
			break;
		img_no++;
	}
	img_no = 0;
	dp = opendir( dir1.c_str() );
	while (dirp = readdir( dp ))
	{
		img_no++;
		filepath = dir1 + "/" + dirp->d_name;
		img = imread(filepath);
		if(!img.data)
			continue;
		resize(img,img,Size(128,128),0,0,INTER_LINEAR );
		vector<Point> locations;
		hog.compute(img, featureVector, winStride, trainingPadding, locations);
		File << "-1";
		for(i=0;i<featureVector.size();i++)
			File <<" "<< i +  1 << ":"<< featureVector[i];
		File << endl;
		if(img_no > 30)
			break;
	}
	File.flush();
	File.close();*/
	printf("Calling SVMlight\n");
	SVMlight::getInstance()->read_problem(const_cast<char*> (featuresFile.c_str()));
	SVMlight::getInstance()->train(); // Call the core libsvm training procedure
	printf("Training done, saving model file!\n");
	SVMlight::getInstance()->saveModelToFile(svmModelFile);

	printf("Generating representative single HOG feature vector using svmlight!\n");
	static vector<float> descriptorVector;
	static vector<unsigned int> descriptorVectorIndices;
	// Generate a single detecting feature vector (v1 | b) from the trained support vectors, for use e.g. with the HOG algorithm
	SVMlight::getInstance()->getSingleDetectingVector(descriptorVector, descriptorVectorIndices);
	// And save the precious to file system
	saveDescriptorVectorToFile(descriptorVector, descriptorVectorIndices, descriptorVectorFile);
	// </editor-fold>

	// <editor-fold defaultstate="collapsed" desc="Test detecting vector">
	descriptorVector.pop_back();
	cout << descriptorVector.size() << endl;
	static vector<float> desc(descriptorVector.begin(),descriptorVector.end());
	cout << desc.size() << endl;
	hog.setSVMDetector();

	vector<Rect> found;
	int groupThreshold = 2;
	Size padding(Size(32, 32));
	Size winStride(Size(8, 8));
	double hitThreshold = 0.; // tolerance
	string im;
	while(1){
	cin >> im;	
	img = imread(im);
	hog.detectMultiScale(img, found, hitThreshold, winStride, padding, 1.05, groupThreshold);

	vector<Rect> found_filtered;
	size_t  j;
	for (i = 0; i < found.size(); ++i) {
		Rect r = found[i];
		for (j = 0; j < found.size(); ++j)
			if (j != i && (r & found[j]) == r)
				break;
		if (j == found.size())
			found_filtered.push_back(r);
	}
	for (i = 0; i < found_filtered.size(); i++) {
		Rect r = found_filtered[i];
		rectangle(img, r.tl(), r.br(), Scalar(64, 255, 64), 3);
	}
	imshow("result",img);
	cvWaitKey(1000);
	}
	return 0;
}
