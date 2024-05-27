#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
//#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/math/special_functions/round.hpp>
#include <pcl/surface/mls.h>        //最小二乘法平滑处理类定义头文件

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/surface/mls.h>        //最小二乘法平滑处理类定义头文件
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/grid_projection.h>
#include <pcl/filters/voxel_grid.h>
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
	//文件句柄  
	__int64 hFile = 0;
	//文件信息  
	struct __finddata64_t  fileinfo;  //很少用的文件信息读取结构
	string p;  //string类很有意思的一个赋值函数:assign()，有很多重载版本
	if ((hFile = _findfirst64(p.assign(path).append("/*.pcd").c_str(), &fileinfo)) == -1)
	{
		cout << "No file is found\n" << endl;
	}
	else
	{
		do
		{
			files.push_back(p.assign(path).append("/").append(fileinfo.name));
		} while (_findnext64(hFile, &fileinfo) == 0);  //寻找下一个，成功返回0，否则-1
		_findclose(hFile);
	}
}


int main(int argc, char** argv)
{
	vector<string> files;
	char* filePath = "...";
	////获取该路径下的所有文件  
	getFiles(filePath, files);
	char str[30];
	int size = files.size();
	for (int i = 0; i < size; i++)
	{
		pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloudOrign(new pcl::PointCloud<pcl::PointXYZRGBL>);
		pcl::console::TicToc time1;
		time1.tic();
		pcl::PCDReader reader;
		// Replace the path below with the path where you saved your file;
		string path = files[i];
		reader.read(path, *cloudOrign); // Remember to download the file first!
		cout << i << endl;
		// 扩充十倍
		for (int j = 0; j <= 9; ++j)
		{
			//计算法线
			pcl::console::TicToc time;
			time.tic();
			cout << "->正在进行随机下采样..." << endl;
			pcl::RandomSample<pcl::PointXYZRGBL> rs;	//创建滤波器对象
			pcl::PointCloud<PointXYZRGBL>::Ptr cloud_sub(new pcl::PointCloud<pcl::PointXYZRGBL>);	//随机下采样点云
			rs.setInputCloud(cloudOrign);
			rs.setSeed(j);
			rs.setSample(1024);					//设置下采样点云的点数
			rs.filter(*cloud_sub);
			cout << "->随机下采样用时：" << time.toc() / 1000 << " s" << endl;
			string::size_type pidx = path.rfind('.', path.length());
			string filename = path.substr(35, pidx);
			string path_b = "D:\\cpp_project\\dadou\\zhenshi\\RS\\RS_1024_END\\" + filename + std::to_string(j) + ".txt";


			savePCDFile<pcl::PointXYZRGBL>(path_b, *cloud_sub);
		}
	}

	return (0);
}