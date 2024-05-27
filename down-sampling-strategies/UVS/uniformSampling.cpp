#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
//#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/math/special_functions/round.hpp>
#include <pcl/surface/mls.h>        //��С���˷�ƽ�������ඨ��ͷ�ļ�

#include <pcl/point_types.h>
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

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/random_sample.h>
#include <pcl/console/time.h>
#include <pcl/filters/uniform_sampling.h>
#include <string>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/visualization/pcl_visualizer.h>

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
		// Replace the path below with the path where you saved your file;
		string path = files[i];
		reader.read(path, *cloudOrign); // Remember to download the file first!

		pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_end(new pcl::PointCloud<pcl::PointXYZL>);
		pcl::PointCloud<pcl::PointXYZL>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZL>);
		pcl::PointXYZL point;
		for (int i = 0; i < cloudOrign->size(); i++) {
			point.x = cloudOrign->points[i].x;
			point.y = cloudOrign->points[i].y;
			point.z = cloudOrign->points[i].z;
			point.label = cloudOrign->points[i].label;
			cloud->push_back(point);
		}

		//���㷨��
		pcl::UniformSampling<PointXYZL> rs;	//�����˲�������
		pcl::console::TicToc time;
		time.tic();
		cout << "->���ڽ����²���..." << endl;
		pcl::PointCloud<PointXYZL>::Ptr cloud_sub(new pcl::PointCloud<pcl::PointXYZL>);
		rs.setInputCloud(cloud);				//���ô��˲�����
		rs.setRadiusSearch(1.6f);				//�������ش�С
		rs.filter(*cloud_sub);
		cout << "->�²�����ʱ��" << time.toc() / 1000 << " s" << endl;

		string::size_type pidx = path.rfind('.', path.length());
		string filename = path.substr(31, pidx);
		string path_b = "..." + filename + ".txt";  // �����ַ


		savePCDFile<pcl::PointXYZL>(path_b, *cloud_sub); //Ĭ�϶����Ʒ�ʽ����

	}

	return (0);
}