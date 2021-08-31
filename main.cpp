#include <iostream>
#include <iomanip> 
#include <string>
#include <vector>
#include<chrono>
#include <fstream>
#include <thread>
#include <mutex>
#include <GL/glut.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/io/vtk_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

static const GLfloat lightpos[] = { 0.0, 0.0, 1.0, 0.0 }; 
static const GLfloat lightcol[] = { 1.0, 1.0, 1.0, 1.0 }; 
static const GLfloat lightamb[] = { 0.1, 0.1, 0.1, 1.0 };
std::mutex m;
std::chrono::system_clock::time_point  start, end; 

Eigen::Matrix3d cmtx;
Eigen::Matrix4d tmtx;

pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PolygonMesh triangles;

pcl::PointCloud<pcl::Normal>::Ptr surface_normals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud (cloud);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    ne.setSearchMethod (tree);

    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

    ne.setRadiusSearch (0.5);

    ne.compute (*cloud_normals);

    return cloud_normals;
}

void Generate_mesh()
{
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal> ());
    cloud_normals = surface_normals(cloud);

    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields (*cloud, *cloud_normals, *cloud_with_normals);

    pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
    tree2->setInputCloud (cloud_with_normals);

    pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;

    gp3.setSearchRadius (3.0);

    gp3.setMu (7.0);
    gp3.setMaximumNearestNeighbors (30);
    gp3.setMaximumSurfaceAngle(M_PI/4);
    gp3.setMinimumAngle(M_PI/18);
    gp3.setMaximumAngle(2*M_PI/2);
    gp3.setNormalConsistency(false);
    std::lock_guard<std::mutex> lock(m);
    gp3.setInputCloud (cloud_with_normals);
    gp3.setSearchMethod (tree2);
    gp3.reconstruct (triangles);

    std::vector<int> parts = gp3.getPartIDs();
    std::vector<int> states = gp3.getPointStates();
}

void read_parm(std::string cparm, Eigen::Matrix3d &cmtx, Eigen:: Matrix4d &tmtx)
{
  cv::Mat cmat, tmat;
  
  cv::FileStorage fs(cparm, cv::FileStorage::READ);
  fs["camera_matrix"] >> cmat;
  fs["trans_matrix"] >> tmat;

  cv::cv2eigen(cmat, cmtx);
  cv::cv2eigen(tmat, tmtx);
  Eigen::Matrix4d trans;
  trans <<  
    0.,-1., 0., 0, 
    0., 0,  -1, 0,
    1, 0., 0, 0, 
    0, 0, 0, 1;
  tmtx = tmtx*trans;  

}

void read_point_cloud(){
  std::lock_guard<std::mutex> lock(m);
  pcl::io::loadPCDFile ("data/test.pcd", *cloud);  
  pcl::VoxelGrid<pcl::PointXYZ> voxel;
  voxel.setLeafSize(0.2f, 0.2f, 0.2f);
  voxel.setInputCloud(cloud);
  voxel.filter(*cloud);
  pcl::transformPointCloud(*cloud, *cloud, tmtx);

}


void Mesh_texture()
{
  //Generate_mesh();
  for(auto &tri: triangles.polygons)
  {
    glBegin(GL_TRIANGLES);

    for (int i = 0; i < 3; ++i)
    {
      pcl::PointXYZ pt;
      Eigen::MatrixXd pn(3,1);
      Eigen::MatrixXd uv(3,1);
      
      pt = cloud->points[tri.vertices[i]];
      pn(0,0) = pt.x/pt.z;
      pn(1,0) = pt.y/pt.z;
      pn(2,0) = 1.0;
      uv = cmtx*pn;

      glTexCoord2d(uv(0,0)/1280, (720-uv(1,0))/720);
      glVertex3d(pt.x, -pt.y, -pt.z);
      
    }
      glEnd();
  } 
}


static void init(void)
{
  cv::Mat img = cv::imread("data/test.png");
  cv::flip(img, img, 0); 
  cv::cvtColor(img, img,cv::COLOR_BGR2RGB);

  glClear(GL_COLOR_BUFFER_BIT);

  glPixelStorei(GL_UNPACK_ALIGNMENT, 3);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.cols, img.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, img.data);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  
  glClearColor(0.3, 0.3, 1.0, 0.0);
  glEnable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
  
  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  //glLightfv(GL_LIGHT0, GL_DIFFUSE, lightcol);
  //glLightfv(GL_LIGHT0, GL_SPECULAR, lightcol);
  //glLightfv(GL_LIGHT0, GL_AMBIENT, lightamb);
}

static void scene(void)
{
  static const GLfloat color[] = { 1.0, 1.0, 1.0, 1.0 }; 
  init(); 
  glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, color);
  glEnable(GL_TEXTURE_2D);

  glNormal3d(0.0, 0.0, 1.0);

  Mesh_texture();
  glDisable(GL_TEXTURE_2D);
}



#include "trackball.h" 

static void display(void)
{
  start = std::chrono::system_clock::now();
  read_point_cloud();
  Generate_mesh();
  
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  
  glLightfv(GL_LIGHT0, GL_POSITION, lightpos);
  
  glTranslated(0.0, 0.0, 0.0);
  
  glMultMatrixd(trackballRotation());
  
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  scene();
  
  glutSwapBuffers();
}

static void resize(int w, int h)
{
  trackballRegion(w, h);
  
  glViewport(0, 0, w, h);
  
  glMatrixMode(GL_PROJECTION);
  
  glLoadIdentity();
  
  gluPerspective(60.0, (double)w / (double)h, 1.0, 100.0);
}

static void idle(void)
{
  glutPostRedisplay();
}

static void mouse(int button, int state, int x, int y)
{
  switch (button) {
  case GLUT_LEFT_BUTTON:
    switch (state) {
    case GLUT_DOWN:
      trackballStart(x, y);
      break;
    case GLUT_UP:
      trackballStop(x, y);
      break;
    default:
      break;
    }
    break;
    default:
      break;
  }
}

static void motion(int x, int y)
{
  end = std::chrono::system_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
  std::cout << 1000.0/elapsed<< std::endl;
  trackballMotion(x, y);
}

static void keyboard(unsigned char key, int x, int y)
{
  switch (key) {
  case 'q':
  case 'Q':
  case '\033':
    exit(0);
  default:
    break;
  }
}

int main(int argc, char *argv[])
{

  read_parm("data/calibration_params.yml",cmtx ,tmtx);
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
  glutCreateWindow(argv[0]);
  
  glutDisplayFunc(display);
  glutReshapeFunc(resize);
  glutIdleFunc(idle);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
  glutKeyboardFunc(keyboard);
  //init();
  glutMainLoop();
  return 0;
}
