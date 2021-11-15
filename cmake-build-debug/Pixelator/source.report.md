## Курсовая работа по дисциплине Обработка изображений
автор: Верещагин А.В.
дата: 2021-11-14T23:35:10

### Текст программы

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;
//#if _DEBUG
//#pragma comment(lib,"opencv_world341d.lib")
//#else
#pragma comment(lib,"opencv_world341.lib")
#

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

    //int pixel_size = 10;
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
    imshow("qu",img_quantized);


//
//    Mat resize_up;
//    // resize back
//    resize(img_quantized, resize_up, Size(img_quantized.cols*pixel_size, img_quantized.rows*pixel_size), INTER_AREA);
//    imshow("qua",resize_up);
//
//    Mat final =  drawgrid(resize_up, pixel_size);
//    imshow("res",final);


    waitKey(0);
    getchar();
    destroyAllWindows();


    return 0;
}

```
