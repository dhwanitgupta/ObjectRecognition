#include <stdio.h>
#include <iostream>
#include <dirent.h>
#include <unistd.h>
#include<iterator>
#include<vector>
#include <sys/stat.h>
#include <sys/types.h>
#include<stdlib.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
using namespace cv;
using namespace std;
long double get_sum(Mat a,Mat b,int m){
	int i;
	long double dis = 100000000;
	//	normalize(b, b, 0, m, NORM_MINMAX, -1, Mat() );
	//	normalize(a, a, 0, m, NORM_MINMAX, -1, Mat() );
	//	cout << a.rows << " " << a.cols << endl;
	//	cout << b.rows << " " << b.cols << endl;
	for(i=0;i<m;i++){
		dis += abs((a.at<double>(0,i) -b.at<double>(0,i))) ;
//		cout << a.at<double>(0,i) << " " << b.at<double>(0,i) << endl;
//		cout << dis << " " ;
	}
//	cout << endl;
	//	cout << "GG " << endl;
//	cout << dis << " ";
	if(dis != dis)
		return 0;
	return dis;
}
int main(){
	Mat train,vocab;
	FileStorage fs("training_descriptors.yml", FileStorage::READ);
	fs["training_descriptors"] >> train;

	FileStorage fs1("vocabulary_color_1000.yml", FileStorage::READ);
	fs1["vocabulary"] >> vocab;

	int k = vocab.rows;
	int discriptor_length = vocab.cols;


	cout << train.rows << " " << train.cols << endl;
	cout << k << " " << discriptor_length << endl;
	int histogram[k];
	int i,j;

	for(i=0;i<k;i++)
		histogram[i] = 0;

	long double min_dist = 123123;
	long double dist;
	int word;
	Mat train_row(1,train.cols,train.type());
	Mat vocab_row(1,vocab.cols,vocab.type());
					
	long double x = 0, x1=0;
	char c[100];
	Mat img_desc;
	int count=0;
	FileStorage fs_img("image_descriptors.yml",FileStorage::READ);
	sprintf(c,"img%d",count++);
	fs_img[string(c)] >> img_desc;
	int xi = 0;
	int image_hist[k];
	for(i=0;i<k;i++)
		image_hist[i] = 0;

	for(i=1;i<train.rows;i++){
		if(img_desc.rows == xi){ 
			sprintf(c,"img%d",count++);
			fs_img[string(c)] >> img_desc;
			for(int si=0;si<k;si++)
				image_hist[si] = 0;
		}
		train.row(i).assignTo(train_row,train.type());
		min_dist = 123231392;
		for(j=0;j<k;j++){
			vocab.row(j).assignTo(vocab_row,vocab.type());
			if(i == 1 && j == 0){
				cout << train_row << endl;
				cout << vocab_row << endl;
			}	
			dist = get_sum(vocab_row,train_row,discriptor_length);
			if(dist < min_dist){
				min_dist = dist;
				word = j;
			}
		}
		histogram[word]++;	
	}
	for(i=0;i<k;i++)
		cout<<histogram[i] << " ";
	cout << endl;

	FileStorage fs_doc("documentVector.yml", FileStorage::WRITE);
	int zz=0;
	/*while(1)
	{
		sprintf(c,"img%d",count++);
		fs_img[string(c)] >> img_desc;
		if(img_desc.rows == 0)
			break;
	//	cout<<img_desc.rows<< " " << img_desc.cols << endl;	
		
	//	static int histimage[300];			//histogram for every image
		for(i=0;i<k;i++)
			histogram[i] = 0;
		
		
		for(i=1; i<116; i++)			//compute histogram (document vector)
		{
			train.row(i).assignTo(train_row,img_desc.type());
			min_dist = 123231392;
			for(j=0;j<k;j++){
				vocab.row(j).assignTo(vocab_row,vocab.type());
				if(i == 1 && j == 0){
					cout << train_row << endl;
					cout << vocab_row << endl;
				}
				dist = get_sum(vocab_row,train_row,discriptor_length);
				if(dist < min_dist){
					min_dist = dist;
					word = j;
				}
			}
//			if(word != 2) cout<<i<<endl;
			histogram[word]++;
		}
//		cout << "i = " << i << endl;
		for(i=0;i<k;i++)
			cout<< histogram[i] << " " ;
		cout << endl;
		vector<int> final;
		for(i=0; i<k; i++)
			final.push_back(histogram[i]);
		if(img_desc.rows > 0)
		{
			Mat w(final,true);
		//	cout << w;
		fs_doc << string(c) << w;
		}
	}*/
	fs_doc.release();
	return 0;
}
