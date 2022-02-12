## Курсовая работа по дисциплине Обработка изображений
автор: Верещагин А.В.
дата: 2022-02-12T17:35:48

### Текст программы

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;
//#if _DEBUG
//#pragma comment(lib,"opencv_world341d.lib")
//#else
//#pragma comment(lib,"opencv_world341.lib")
//#

Mat colorq(Mat ocv, int cluster_count){

// convert to float & reshape to a [3 x W*H] Mat
//  (so every pixel is on a row of it's own)

    Mat data;
    ocv.convertTo(data,CV_32F);
    data = data.reshape(1,data.total());

// do kmeans
    Mat labels, centers;
    kmeans(data, cluster_count, labels, TermCriteria(2, 10, 1.0), 3,
           KMEANS_PP_CENTERS, centers);

// reshape both to a single row of Vec3f pixels:
    centers = centers.reshape(3,centers.rows);
    data = data.reshape(3,data.rows);

// replace pixel values with their center value:
    Vec3f *p = data.ptr<Vec3f>();
    for (size_t i=0; i<data.rows; i++) {
        int center_id = labels.at<int>(i);
        p[i] = centers.at<Vec3f>(center_id);
    }

// back to 2d, and uchar:
    ocv = data.reshape(3, ocv.rows);
    ocv.convertTo(ocv, CV_8U);
    return ocv;
}

Mat bgrThreshold_white(Mat img){
    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++)
            if (img.at<Vec3b>(i, j)[0] > 210  && img.at<Vec3b>(i, j)[1] >  210 &&img.at<Vec3b>(i, j)[2] >  210)
            {
                img.at<Vec3b>(i, j)[0] = 255;
                img.at<Vec3b>(i, j)[1] = 255;
                img.at<Vec3b>(i, j)[2] = 255;
            }
    return img;
}

Mat drawgrid( Mat mat_img,int stepSize ){

    int width = mat_img.size().width;
    int height = mat_img.size().height;

    for (int i = 0; i<height; i += stepSize)
        cv::line(mat_img, Point(0, i), Point(width, i), cv::Scalar(0, 0, 0));

    for (int i = 0; i<width; i += stepSize)
        cv::line(mat_img, Point(i, 0), Point(i, height), cv::Scalar(0, 0, 0));
    return mat_img;
}

Mat makebig(Mat img, int pixsize) {
    Mat big;
    int new_width = img.size().width*pixsize;
    int new_height = img.size().height*pixsize;
    big.push_back(Mat(new_height,new_width,CV_8UC3,Vec<uchar, 3>(255,255,255)));
    for (int i = 0; i < new_height; i++)
        for (int j = 0; j < new_width; j++)
            big.at<Vec3b>(i, j) = img.at<Vec3b>(i/pixsize, j/pixsize);
    return big;
}
Mat numerator(Mat sour, Mat nums){
    int l;
    for (int i = 0; i < nums.size().height; i++) {
        for (int j = 0; j < nums.size().width; j++) {
            l = nums.at<int>(i, j);
            if (sour.at<Vec3b>(i * 20 + 4, j * 20 + 14) != Vec<uchar, 3>(255,255,255)) {
                cv::putText(sour, //target image
                            to_string(nums.at<int>(i, j)), //text
                            cv::Point(j * 20 + 4, i * 20 + 14), //top-left position
                            cv::FONT_HERSHEY_DUPLEX,
                            0.3,
                            CV_RGB(0, 0, 0), //font color
                            1);
            }
        }
    }
    return sour;
}
int printe(Mat nums){
    int l;
    for (int j = 0; j < nums.size().width; j++) {
        for (int i = 0; i < nums.size().height; i++){
            l = nums.at<int>(i, j);
            }
        }
    }

Mat kroswording1(Mat img){
    Mat ebala;
    Mat e1;
    e1.push_back(Mat(img.size().height,img.size().width,CV_8UC1, int(0)));
    ebala.push_back(Mat(img.size().height,img.size().width,CV_8UC3,Vec<uchar, 3>(255,255,255)));
    int nums[img.size().height][img.size().width];
    int x =0;
    int y =0;
    Vec<uchar, 3> tem;
    for (int i = 0; i < img.size().height; i++) {
        for (int j = 0; j < img.size().width; j++){
            if (img.at<Vec3b>(i, j) != Vec<uchar, 3>(255, 255, 255)){
                if (img.at<Vec3b>(i, j) == tem){
                    e1.at<int>(y,x-1) = e1.at<int>(y,x-1) +1;
                    nums[y][x-1] = nums[y][x-1]+1;
                }
                if (img.at<Vec3b>(i, j) != tem) {
                    ebala.at<Vec3b>(y, x) = img.at<Vec3b>(i, j);
                    e1.at<int>(y,x)=1;
                    nums[y][x]=1;
                    x++;

                }


                tem = img.at<Vec3b>(i, j);
            }
        //tem=Vec<uchar, 3>(255, 255, 255);

        }
        tem = Vec3b(3,2,3);
        x=0;
        y++;
    }
    Mat fu = makebig(ebala,20);
    for (int i = 0; i < img.size().height; i++) {
        for (int j = 0; j < img.size().width; j++) {
            if (fu.at<Vec3b>(i * 20 + 4, j * 20 + 14) != Vec<uchar, 3>(255,255,255)) {
                cv::putText(fu, //target image
                            to_string(nums[i][j]), //text
                            cv::Point(j * 20 + 5, i * 20 + 12), //top-left position
                            cv::FONT_HERSHEY_DUPLEX,
                            0.3,
                            CV_RGB(0, 0, 0), //font color
                            1);
            }
        }
    }
    //fu = numerator(fu,e1);
    return fu;
}

Mat skl1(Mat img1, Mat img2){

    // Get dimension of final image
    int rows = max(img1.rows, img2.rows);
    int cols = img1.cols + img2.cols;

    Mat3b res(rows, cols, Vec3b(255,255,255));

    // Copy images in correct position
    img1.copyTo(res(Rect(0, 0, img1.cols, img1.rows)));
    img2.copyTo(res(Rect(img1.cols, 0, img2.cols, img2.rows)));
    return res;
}
Mat skl2(Mat img1, Mat img2){

    // Get dimension of final image
    int rows = (img1.rows + img2.rows);
    int cols = max(img1.cols, img2.cols);

    Mat3b res(rows, cols, Vec3b(255,255,255));

    // Copy images in correct position
    img1.copyTo(res(Rect(0, 0, img1.cols, img1.rows)));
    img2.copyTo(res(Rect(0, img1.rows, img2.cols, img2.rows)));
    return res;
}

Mat kroswording2(Mat img){
    Mat ebala;
    Mat e2;
    ebala.push_back(Mat(img.size().height,img.size().width,CV_8UC3,Vec<uchar, 3>(255,255,255)));
    int nums[img.size().height][img.size().width];
    e2.push_back(Mat(img.size().height,img.size().width,CV_8UC1, int(0)));
    int x =0;
    int y =0;
    int l;
    l=e2.at<int>(1,1);
    Vec<uchar, 3> tem;
    for (int i = 0; i < img.size().width; i++) {
        for (int j = 0; j < img.size().height; j++) {
            if (img.at<Vec3b>(j, i) != Vec<uchar, 3>(255, 255, 255)){
                if (img.at<Vec3b>(j, i) == tem){
                    nums[x-1][y] =  e2.at<int>(x-1,y) +1;
                    e2.at<int>(x-1, y) = e2.at<int>(x-1,y) +1;
                }
                if (img.at<Vec3b>(j, i) != tem) {
                    ebala.at<Vec3b>(x, y) = img.at<Vec3b>(j, i);
                    nums[x][y]=1;
                    e2.at<int>(x, y) = 1;
                    x++;
                }
                tem = img.at<Vec3b>(j, i);
            }

        }
        l=e2.at<int>(1,1);
        tem = Vec3b(3,2,3);
        l=e2.at<int>(0,0);
        y++;
        x=0;
    }
    int z = printe(e2);
    Mat fu = makebig(ebala,20);
    for (int i = 0; i < img.size().height; i++) {
        for (int j = 0; j < img.size().width; j++) {
            if (fu.at<Vec3b>(i * 20 + 4, j * 20 + 14) != Vec<uchar, 3>(255,255,255)) {
                l=e2.at<int>(1,1);
                cv::putText(fu, //target image
                            to_string(nums[i][j]), //text
                            cv::Point(j * 20 + 5, i * 20 + 12), //top-left position
                            cv::FONT_HERSHEY_DUPLEX,
                            0.3,
                            CV_RGB(0, 0, 0), //font color
                            1);
            }
        }
    }

    //fu = numerator(fu,e2);
    return fu;
}
int main()
{

    int colors_count=3;
    int down_height = 50;
    cout << "Задайте желаемое количество строк для кроссворда" << endl;
    cout << "20 -маленький, 50 - средний, 100 - большой" << endl;
    cin >> down_height ;
    cout << "Задайте желаемое количество цветов кроссворда, рекомендуется от 3 до 8" << endl;
    cin >> colors_count;
    std::string path = "/home/alex/Загрузки/3.jpg";
    std::string pa;
    cout << "Введите путь до картинки или /home/alex/Загрузки/3.jpg" << endl;
    cin>> path;

    Mat image = imread(path);

    if (image.empty()) {
        cout << "Could not read image" << endl;
        return 0;
    }

    int pixel_size = 20;
    int down_width = image.cols / (image.rows / down_height);
    Mat resize_down;
    resize(image, resize_down, Size(down_width, down_height), INTER_NEAREST);
    //бинаризация порог 200
//    imshow("1", resize_down);
    Mat img_gray;
    cvtColor(resize_down, img_gray, COLOR_BGR2GRAY);
  //  imshow("2", img_gray);
    Mat inverted_binary_image;
    //Mat thresh;
    threshold(img_gray, inverted_binary_image, 200, 255, THRESH_BINARY_INV);
    //imshow("3", inverted_binary_image);

    //bitwise_not(thresh, inverted_binary_image);
    //изменение размера

    //контуры
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(inverted_binary_image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<Point> contour = contours[0];
    int start_col, end_col, start_row, end_row;
    Point extLeft  = *min_element(contour.begin(), contour.end(),
                                  [](const Point& lhs, const Point& rhs){
                                      return lhs.x < rhs.x;
                                  });
    start_col = extLeft.x;
    Point extRight = *max_element(contour.begin(), contour.end(),
                                  [](const Point& lhs, const Point& rhs) {
                                      return lhs.x < rhs.x;
                                  });
    end_col = extRight.x;
    Point extTop   = *min_element(contour.begin(), contour.end(),
                                  [](const Point& lhs, const Point& rhs) {
                                      return lhs.y < rhs.y;
                                  });
    start_row = extTop.y;
    Point extBot   = *max_element(contour.begin(), contour.end(),
                                  [](const Point& lhs, const Point& rhs) {
                                      return lhs.y < rhs.y;
                                  });
    end_row = extBot.y;

    // draw contours on the original image
//   Mat image_copy = resize_down.clone();
//   drawContours(image_copy, contours, -1, Scalar(0, 255, 0), 2);
//   imshow("None approximation", image_copy);
//    imshow("hh",img_gray);
//    imshow("None ", thresh);
//    waitKey();
//    cout<<start_row<< end_row<< start_col<< end_col<<endl;
//    imshow("Resized Down by defining height and width", resize_down);

    //кроп
    Mat crop = resize_down(Range(start_row, end_row), Range(start_col, end_col));
//    imshow("Croped mage", crop);
//    waitKey();
    Mat img_quantized;
    img_quantized = colorq(crop,colors_count+1);
    //imshow("4", img_quantized);

    img_quantized = bgrThreshold_white(img_quantized);
    imwrite("pixelated.png",img_quantized );
    //imshow("qu",img_quantized);

    Mat z = kroswording1(img_quantized);
    Mat x = kroswording2(img_quantized);

//    // уведичиваем матрицу назад применяя интерполяцию по площади
    Mat resize_up = makebig(img_quantized, pixel_size);
//    imshow("aaa", z);
    Mat y = skl1(resize_up, z);
    Mat ya = skl2(y,x);

    //resize(img_quantized, resize_up, Size(img_quantized.cols*pixel_size, img_quantized.rows*pixel_size), INTER_NEAREST);
    //imshow("qua",ya);
    // рисуем сетку
    Mat final =  drawgrid(ya, pixel_size);
    imshow("result",final);


    waitKey(0);
    getchar();
    destroyAllWindows();


    return 0;
}

```
