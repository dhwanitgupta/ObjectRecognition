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
static string svmModelFile = "genfiles/svmlightmodel.dat";
int main(){
	initModule_nonfree();

	string featuresFile("Hogfeatures");
	fstream File;
	File.open(featuresFile.c_str(), ios::out);
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
	while (dirp = readdir( dp ))
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
	}
	dp = opendir( dir1.c_str() );
	while (dirp = readdir( dp ))
	{
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
	}
	File.flush();
	File.close();
	printf("Calling SVMlight\n");
	SVMlight::getInstance()->read_problem(const_cast<char*> (featuresFile.c_str()));
	SVMlight::getInstance()->train(); // Call the core libsvm training procedure
	printf("Training done, saving model file!\n");
	SVMlight::getInstance()->saveModelToFile(svmModelFile);
	return 0;
}
