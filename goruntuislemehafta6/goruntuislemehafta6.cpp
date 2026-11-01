#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

int main()
{
    Mat resim = imread("images2.jpg", IMREAD_COLOR);

    if (resim.empty())
    {
        cout << "images2.jpg bulunamadi!" << endl;
        return -1;
    }

    // 1. Griye donustur.
    Mat gri;
    cvtColor(resim, gri, COLOR_BGR2GRAY);

    // 2. Canny ile kenarlari bul.
    Mat kenarlar;
    Canny(gri, kenarlar, 50, 150, 3);

    // 3. Kenar goruntusune median uygula.
    Mat medianKenarlar;
    medianBlur(kenarlar, medianKenarlar, 3);

    cout << "Canny kenar pikseli: "
        << countNonZero(kenarlar) << endl;

    cout << "Median sonrasi kenar pikseli: "
        << countNonZero(medianKenarlar) << endl;

    // 4. Hazir HoughCircles fonksiyonunu uygula.
    // Bu degerler gonderdigin 554x554 para resmi icin
    // baslangic parametreleridir.
    double merkezlerArasiMesafe = 70;
    double cannyUstEsik = 100;
    double merkezOyEsigi = 15;

    int minimumYaricap = 30;
    int maksimumYaricap = 70;

    vector<Vec3f> cemberler;

    HoughCircles(
        medianKenarlar,
        cemberler,
        HOUGH_GRADIENT,
        1.0,                       // Oylama cozunurlugu orani
        merkezlerArasiMesafe,
        cannyUstEsik,              // Dahili Canny ust esigi
        merkezOyEsigi,
        minimumYaricap,
        maksimumYaricap
    );

    // 5. Cemberleri orijinal resmin kopyasina ciz.
    Mat sonuc = resim.clone();

    cout << "\nBulunan cember sayisi: "
        << cemberler.size() << endl;

    for (size_t i = 0; i < cemberler.size(); i++)
    {
        Point merkez(
            cvRound(cemberler[i][0]),
            cvRound(cemberler[i][1])
        );

        int yaricap = cvRound(cemberler[i][2]);

        // Cember cevresi: kirmizi.
        circle(
            sonuc,
            merkez,
            yaricap,
            Scalar(0, 0, 255),
            2,
            LINE_AA
        );

        // Merkez: yesil.
        circle(
            sonuc,
            merkez,
            3,
            Scalar(0, 255, 0),
            FILLED,
            LINE_AA
        );

        // Cember numarasi.
        putText(
            sonuc,
            to_string(i + 1),
            Point(merkez.x + 5, merkez.y - 5),
            FONT_HERSHEY_SIMPLEX,
            0.6,
            Scalar(255, 0, 0),
            2
        );

        cout << "Cember " << i + 1
            << ": merkez=("
            << merkez.x << ", " << merkez.y << ")"
            << " yaricap=" << yaricap << endl;
    }

    if (cemberler.empty())
    {
        cout << "\nCember bulunamadi." << endl;
        cout << "Median ince kenarlari silmis olabilir."
            << endl;
        cout << "Canny ve median goruntulerini kontrol et."
            << endl;
    }

    // 6. Sonuclari kaydet.
    imwrite("gorev3_gri.png", gri);
    imwrite("gorev3_canny.png", kenarlar);
    imwrite("gorev3_median.png", medianKenarlar);
    imwrite("gorev3_hough_circles.png", sonuc);

    // 7. Goster.
    imshow("Orijinal", resim);
    imshow("Canny", kenarlar);
    imshow("Canny sonrasi Median", medianKenarlar);
    imshow("HoughCircles - Kirmizi Cemberler", sonuc);

    waitKey(0);
    return 0;
}