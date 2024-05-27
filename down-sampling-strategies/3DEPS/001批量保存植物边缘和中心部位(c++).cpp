#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
//#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/math/special_functions/round.hpp>
#include <pcl/surface/mls.h>        //��С���˷�ƽ�������ඨ��ͷ�ļ�

#include <pcl/io/pcd_io.h>
#include <pcl/surface/mls.h>        //��С���˷�ƽ�������ඨ��ͷ�ļ�
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/grid_projection.h>
#include <iostream>
#include <string.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <time.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/features/boundary.h>

using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;
using namespace std;
void getFiles(string path, vector<string>& files)
{
	//�ļ����  
	__int64 hFile = 0;
	//�ļ���Ϣ  
	struct __finddata64_t  fileinfo;  //�����õ��ļ���Ϣ��ȡ�ṹ
	string p;  //string�������˼��һ����ֵ����:assign()���кܶ����ذ汾
	if ((hFile = _findfirst64(p.assign(path).append("/*.pcd").c_str(), &fileinfo)) == -1)
	{
		cout << "No file is found\n" << endl;
	}
	else
	{
		do
		{
			files.push_back(p.assign(path).append("/").append(fileinfo.name));
		} while (_findnext64(hFile, &fileinfo) == 0);  //Ѱ����һ�����ɹ�����0������-1
		_findclose(hFile);
	}
}


int main(int argc, char** argv)
{
	vector<string> files;
	char* filePath = "...";
	////��ȡ��·���µ������ļ�  
	getFiles(filePath, files);
	char str[30];
	int size = files.size();
	for (int i = 0; i < size; i++)
	{
		pcl::PointCloud<pcl::PointXYZL>::Ptr cloudOrign(new pcl::PointCloud<pcl::PointXYZL>);
		pcl::PCDReader reader;
		string path = files[i];
		reader.read(path, *cloudOrign); // Remember to download the file first!

		pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_b(new pcl::PointCloud<pcl::PointXYZL>);
		pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_c(new pcl::PointCloud<pcl::PointXYZL>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointXYZ point;
		for (int i = 0; i < cloudOrign->size(); i++) {
			point.x = cloudOrign->points[i].x;
			point.y = cloudOrign->points[i].y;
			point.z = cloudOrign->points[i].z;
			cloud->push_back(point);
		}

		//���㷨��
		pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normEst;
		pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>);
		tree->setInputCloud(cloud);
		normEst.setInputCloud(cloud);
		normEst.setSearchMethod(tree);
		normEst.setKSearch(20);
		normEst.compute(*normals);
		//�жϱ�Ե��
		pcl::PointCloud<pcl::Boundary> boundaries;
		pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> boundEst;
		tree2->setInputCloud(cloud);
		boundEst.setInputCloud(cloud);
		boundEst.setInputNormals(normals);
		boundEst.setSearchMethod(tree2);
		boundEst.setKSearch(20);
		boundEst.setAngleThreshold(M_PI / 2);
		boundEst.compute(boundaries);
		//��ȡ��Ե���������
		cloud_b->width = cloudOrign->points.size();
		cloud_b->height = 1;
		cloud_b->points.resize(cloud_b->width * cloud_b->height);
		//��ȡ�Ǳ�Ե���������
		cloud_c->width = cloudOrign->points.size();
		cloud_c->height = 1;
		cloud_c->points.resize(cloud_c->width * cloud_c->height);
		int j = 0;
		int k = 0;
		for (int i = 0; i < cloudOrign->points.size(); i++)
		{
			if (boundaries.points[i].boundary_point != 0)
			{
				cloud_b->points[j].x = cloudOrign->points[i].x;
				cloud_b->points[j].y = cloudOrign->points[i].y;
				cloud_b->points[j].z = cloudOrign->points[i].z;
				cloud_b->points[j].label = cloudOrign->points[i].label;
				j++;
			}
			else
			{
				cloud_c->points[k].x = cloudOrign->points[i].x;
				cloud_c->points[k].y = cloudOrign->points[i].y;
				cloud_c->points[k].z = cloudOrign->points[i].z;
				cloud_c->points[k].label = cloudOrign->points[i].label;
				k++;
			}
			continue;
		}
		cloud_b->width = j;
		cloud_b->points.resize(cloud_b->width * cloud_b->height);
		cloud_c->width = k;
		cloud_c->points.resize(cloud_c->width * cloud_c->height);
		cout << "********" << i << "********" << endl;
		cout << "ԭʼ����" << cloudOrign->size() << endl;
		cout << "��Ե����" << cloud_b->size() << endl;

		//string::size_type idx = path.rfind('\\', path.length());�ֱ𱣴��Ե���ڲ���
		string path_e = "D:\\cpp_project\\PCL1\\downsampling_ex\\3DEPS\\data_edge&core\\edge\\";
		string path_c = "D:\\cpp_project\\PCL1\\downsampling_ex\\3DEPS\\data_edge&core\\core\\";
		//��ȡ��ԭ���������ļ�����
		string::size_type pidx_e = path.rfind('/', path.length());
		string::size_type pidx_c = path.rfind('.', path.length());
		string filename_e = path.substr(pidx_e + 1, pidx_c);
		//string filename_c = path.substr(0, pidx_c);
		path_e = path_e + filename_e + "_e.txt";
		path_c = path_c + filename_e + "_c.txt";

		savePCDFile<pcl::PointXYZL>(path_e, *cloud_b); //Ĭ�϶����Ʒ�ʽ����
		savePCDFile<pcl::PointXYZL>(path_c, *cloud_c); //Ĭ�϶����Ʒ�ʽ����
	}

	return (0);
}