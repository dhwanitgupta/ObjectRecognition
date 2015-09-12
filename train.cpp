#include <stdio.h>
#include <iostream>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include<stdlib.h>
#include"cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <algorithm>   
using namespace cv;
using namespace std;
int in_check(int data, int ar[8],int pos){
	int i;
	for(i=0;i<pos;i++)
		if(ar[i] == data)
			return 1;

	return 0;
}
int main(){
	string dir = "Caltech_11classes/test1", filepath;
	string dir1 = "Caltech_11classes/test";
	DIR *dp;
	struct dirent *dirp;
	struct stat filestat;

	Mat inverse_index,vocabulary,img_vector;
	FileStorage fs("inverse_index.yml", FileStorage::READ);
	fs["inv_index"] >> inverse_index;
	FileStorage fs1("vocabulary.yml", FileStorage::READ);
	fs1["vocabulary"] >> vocabulary;
	FileStorage fs2("image_vector.yml", FileStorage::READ);


	int no_words = inverse_index.rows;
	int no_images = inverse_index.cols;
	Ptr<DescriptorExtractor > extractor(
			new OpponentColorDescriptorExtractor(
				Ptr<DescriptorExtractor>(new SiftDescriptorExtractor())
				)
			);

	Ptr<FeatureDetector> featureDetector = FeatureDetector::create( "SIFT");
	Ptr<BOWImgDescriptorExtractor> bowExtractor;
	Ptr<DescriptorMatcher> descMatcher = DescriptorMatcher::create( "BruteForce" );
	bowExtractor = new BOWImgDescriptorExtractor( extractor, descMatcher );
	bowExtractor->setVocabulary( vocabulary );
	vector<KeyPoint> keypoints;
	SiftFeatureDetector detector(100);
	int True = 0;
	int False = 0;
	DIR *dp1;
	struct dirent *dirp1;
	struct stat filestat1;
	dp1 = opendir( dir1.c_str() );
	while (dirp1 = readdir( dp1 ))
	{
		Mat img,response_hist;
		filepath = dir1 + "/" + dirp1->d_name;
		if (stat( filepath.c_str(), &filestat )) continue;
		if (S_ISDIR( filestat.st_mode ))         continue;

		img = imread("test_file.jpg");
//		string in_class(filepath,23,3);

		if (!img.data) {
			continue;
		}
	detector.detect(img,keypoints);
	bowExtractor->compute(img, keypoints, response_hist);
	//cout << response_hist << endl;
	int i,j;
	char c[100];
	int wordcount[no_images];
	for(i=0;i<no_images;i++)
		wordcount[i] = 0;
	int count = 0;
	//cout << inverse_index << endl;
	for(i=0;i<response_hist.cols;i++){
		//	cout << "res = " << response_hist.at<int>(0,i) << endl;	
		if(response_hist.at<int>(0,i) > 100){
			//	cout << "word " << i << " " ;
			count++;
			for(j=0;j<no_images;j++){
				if(inverse_index.at<int>(i,j) > 100 ){
					//		cout << j << " ";
					//			cout << inverse_index.at<int>(i,j) << " ";
					wordcount[j]++;
				}
			}
		}
	}
	double min_val = 10000000000;
	double max_val = -1;
	double val;
	int imno,im_no;
	int min_index;
	double top_val[8];
	int index_val[8];
	int pos = 0;
	for(i=0;i<no_images;i++){
		if(wordcount[i] > count/4){
			sprintf(c,"img%d",i);
			fs2[string(c)] >> img_vector;
			val = img_vector.dot(response_hist)/(norm(img_vector) * norm(response_hist));
			if(pos < 8 ){
				if(min_val > val){
					min_val = val;
					min_index = pos;
				}
				top_val[pos] = val;
				index_val[pos] = i;
				pos++;
			}
			else{
				if(min_val < val ){
					top_val[min_index] = val;
					index_val[min_index] = i;
					min_val = 10000000000;
					for(int com = 0 ; com < pos ; com++){
						if(top_val[com] < min_val){
							min_val = top_val[com];
							min_index = com;
						}
					}
				}
			}
			if(val > max_val){
					max_val = val;
					im_no = i;
			}
			//		cout << i << endl;
		}
	}
	dp = opendir( dir.c_str() );
	//cout << im_no << endl;
	//namedWindow("input",-1);
	//imshow("input",img);
	i = 0;
	imno = 0;
	int flag = 0;
	int fin;
	vector<double> sorted_val;
	sorted_val.assign(top_val,top_val+8);
	sort(sorted_val.begin(),sorted_val.end());
	while (dirp = readdir( dp ))
	{
		filepath = dir + "/" + dirp->d_name;
		if (stat( filepath.c_str(), &filestat )) continue;
		if (S_ISDIR( filestat.st_mode ))         continue;

		img = imread(filepath);
		if (!img.data) {
			continue;
		}
		
		if(in_check(imno,index_val,pos) /*|| imno == im_no*/){
			//cout << string(filepath,24,3) << endl;
			//if(out_class == in_class)
			//	flag = 1;
	//		cout<<top_val[i]<<endl;
			for(fin = 0 ; fin < 8 ; fin++)
				if(sorted_val[fin] == top_val[i])
					break;	
	//		cout<< fin <<endl;	
			sprintf(c,"output%d.jpg",7-fin);
		//	namedWindow(c,-1);
		//	imshow(c,img);
		//	cvWaitKey(1000);
			imwrite(c,img);
			i++;
		}
		if( i> pos)
			break;
		imno++;
	}
	break;
	}
	return 0;
}
