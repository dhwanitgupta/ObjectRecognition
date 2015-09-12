#include <stdio.h>
#include <iostream>
#include <dirent.h>
#include <iomanip>      // std::setprecision
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include<stdlib.h>
#include<string.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/nonfree.hpp>
using namespace cv;
using namespace std;
int main(){
	initModule_nonfree();




	string dir = "Caltech_11classes/test1", filepath;
	DIR *dp;
	struct dirent *dirp;
	struct stat filestat;

	dp = opendir( dir.c_str() );

	// detecting keypoints
	SurfFeatureDetector detector(100);
	//FastFeatureDetector detector(1,true);
	vector<KeyPoint> keypoints;	

	// computing descriptors
	//Ptr<DescriptorExtractor > extractor(new SurfDescriptorExtractor());//  extractor;
	Ptr<DescriptorExtractor > extractor(
			new OpponentColorDescriptorExtractor(
				Ptr<DescriptorExtractor>(new SurfDescriptorExtractor())
				)
			);
	Mat training_descriptors(1,extractor->descriptorSize(),extractor->descriptorType());
	Mat img;

	cout << "------- build vocabulary ---------\n";

	cout << "extract descriptors.."<<endl;
	//Rect clipping_rect = Rect(0,120,640,480-120);
	//Mat bg_ = imread("background.png")(clipping_rect),
	Mat img_fg;
	FileStorage fs_img("image_descriptors.yml", FileStorage::WRITE);
	int count = 0;
	char c[100];
	unsigned found;
	const TermCriteria& tc  = TermCriteria(CV_TERMCRIT_ITER, 100, 0.001);
	int retries = 1;
	int train_flags = KMEANS_PP_CENTERS;
	BOWKMeansTrainer bowtrainer(1000,tc,retries,train_flags); 
	while (dirp = readdir( dp ))
	{
		Mat descriptors;
		filepath = dir + "/" + dirp->d_name;
		if (stat( filepath.c_str(), &filestat )) continue;
		if (S_ISDIR( filestat.st_mode ))         continue;
		img = imread(filepath);
		if (!img.data) {
			continue;
		}
		img_fg = img;
		detector.detect(img_fg, keypoints);
		extractor->compute(img, keypoints, descriptors);
		found = string(dirp->d_name).find(".");
		sprintf(c,"img%d",count);
		//	fs_img << string(c)<< descriptors;
		count++;
		//training_descriptors.push_back(descriptors);
		bowtrainer.add(descriptors);
		cout << ".";
	}
	fs_img.release();
	cout << endl;
	closedir( dp );

//	cout << "Total descriptors: " << training_descriptors.rows << endl;

	/*	FileStorage fs("training_descriptors.yml", FileStorage::WRITE);
		fs << "training_descriptors" << training_descriptors;
		fs.release();*/

	cout << "cluster BOW features" << endl;
	Mat vocabulary = bowtrainer.cluster();
	cout << "Vocab Done" <<endl;
	FileStorage fs1("vocabulary.yml", FileStorage::WRITE);
	fs1 << "vocabulary" << vocabulary;
	fs1.release();

	dp = opendir( dir.c_str() );
	cout << vocabulary.rows << " "<<vocabulary.cols<<endl;
	Ptr<FeatureDetector> featureDetector = FeatureDetector::create( "SURF");
	Ptr<DescriptorExtractor> descExtractor = DescriptorExtractor::create( "SURF" );
	Ptr<BOWImgDescriptorExtractor> bowExtractor;
	Ptr<DescriptorMatcher> descMatcher = DescriptorMatcher::create( "BruteForce" );
	bowExtractor = new BOWImgDescriptorExtractor( extractor, descMatcher );
	bowExtractor->setVocabulary( vocabulary );
	vector<KeyPoint> keypoints1;
	Mat temp;
	int k = 1000;
	//	double inv_ind[k][count];
	vector< vector<double> > inv_map;
	int i = 0;
	inv_map.resize(k);
	for(i=0;i<k;i++)
		inv_map[i].resize(count);
	//		for(int j = 0  ; j < count ; j++){
	//			inv_ind[i][j] = 0;
	//	}
	int im_no = 0;
	cout << "img = " << count << endl;
	FileStorage fs2("image_vector.yml", FileStorage::WRITE);
	Mat response_hist;
	while (dirp = readdir( dp ))
	{
		filepath = dir + "/" + dirp->d_name;

		if (stat( filepath.c_str(), &filestat )) continue;
		if (S_ISDIR( filestat.st_mode ))         continue;

		img = imread(filepath);
		//cout << img.rows << " " << img.cols << endl;
		if (!img.data) {
			continue;
		}
		//	cout << keypoints1.size() << " " << filepath << endl;
		detector.detect(img, keypoints1);
		if(keypoints1.size() > 0){
			bowExtractor->compute(img, keypoints1, response_hist);
		}
		sprintf(c,"img%d",im_no);
		fs2 << string(c) << response_hist;
		for(i=0;i<response_hist.cols;i++){
			if( response_hist.at<double>(0,i) != 0){
				//				inv_ind[i][im_no] = response_hist.at<double>(0,i);
				inv_map[i][im_no] = 1;//response_hist.at<double>(0,i);
			}
		}
		im_no++;
	}
	Mat inv_mat(k,count,5);
	for(i=0;i<k;i++){
		//	cout << Mat(inv_map[i],true) << endl;
		for(im_no=0;im_no<count;im_no++)
			inv_mat.row(i).col(im_no) = inv_map[i][im_no];
		//	cout<<endl;
	}
	fs2.release();
	FileStorage fs("inverse_index.yml", FileStorage::WRITE);
	fs << "inv_index" << inv_mat;
	fs.release();
	closedir( dp );
	return 0;
}
