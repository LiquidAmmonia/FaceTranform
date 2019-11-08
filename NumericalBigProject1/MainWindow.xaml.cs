using Microsoft.Win32;
using System;
using System.Drawing;
using System.IO;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Media.Imaging;
using System.Threading.Tasks;

namespace NumericalBigProject1
{
    
    public partial class MainWindow : Window
    {
        //open or save img
        public BitmapImage OriginalImg;
        public BitmapImage CurrentImg;
        public int SaveCounter;
        public byte[,,] OriginalMatrix;
        public BitmapImage FaceImg;
        public byte[,,] FaceMatrix;
        
        //Path
        public string OriginalImgPath;
        public string FaceImgPath;

        //TPS Face Points
        public double[,] OriginalFacePoints;
        public double[,] ChangeFacePoints;

        //interpolate methods
        public int InterpolateMethod;//0 for nearest; 1 for bilinear; 2 for bicubic

        // rotate transform
        public double Rotate;
        public double Radius;

        // distortion transform
        public double DistortionRadus;        

        public class MyMatrix
        {
            public double[,] Value;
            public int Rows;
            public int Cols;
            public MyMatrix(double [,] Val)
            {
                //自定义的matrix类 支持求逆,矩阵乘法,切片等操作
                this.Rows = Val.GetLength(0);
                this.Cols = Val.GetLength(1);
                Value = new double[Rows, Cols];
                //initialize
                for(int i = 0; i < Rows; i++)
                    for(int j = 0; j < Cols; j++)
                        Value[i, j] = Val[i, j];                              
            }
            public MyMatrix(MyMatrix M)
            {
                //复制构造函数
                this.Rows = M.Rows;
                this.Cols = M.Cols;
                this.Value = new double[Rows, Cols];
                for (int i = 0; i < Rows; i++)                
                    for (int j = 0; j < Cols; j++)                    
                        Value[i, j] = M.Value[i, j];                
            }
            public MyMatrix Transpose()
            {
                //矩阵转置
                int r = this.Rows;
                int c = this.Cols;
                double[,] newValue = new double[c, r];
                for(int i = 0; i < r; i++)                
                    for(int j = 0; j < c; j++)                    
                        newValue[j, i] = Value[i, j];               
                var new_M = new MyMatrix(newValue);
                return new_M;
            }
            public void Append(MyMatrix M, int l)
            {
                //矩阵在右边或下边concatenate
                // 0 for col; 1 for row 
                if(l == 0)
                {
                    int c = this.Cols;
                    this.Cols += M.Cols;
                    var newValue = new double[this.Rows, this.Cols];                    
                    for(int i = 0; i < this.Cols; i++)
                    {                        
                        for(int j = 0;j<this.Rows;j++)
                        {
                            if (i < c)
                            {
                                newValue[j, i] = Value[j, i];
                            }
                            else
                            {
                                newValue[j, i] = M.Value[j, i - c];
                            }
                        }                        
                    }
                    Value = null;
                    Value = newValue;
                }
                if(l == 1)
                {
                    int r = this.Rows;
                    this.Rows += M.Rows;
                    var newValue = new double[this.Rows, this.Cols];

                    for (int i = 0; i < this.Cols; i++)
                    {
                        for (int j = 0; j < this.Rows; j++)
                        {
                            if (j < r)
                            {
                                newValue[j, i] = Value[j, i];
                            }
                            else
                            {
                                newValue[j, i] = M.Value[j-r, i];
                            }
                        }
                    }
                    Value = null;
                    Value = newValue;
                }
            }            
            public MyMatrix Cut(int row=1, int col=1)
            {
                //矩阵切片
                //return the col 
                var newMat = new double[row, col];
                for(int i = 0; i < row; i++)
                {
                    for(int j = 0; j < col; j++)
                    {

                        newMat[i, j] = this.Value[i, j];
                    }
                }
                var new_M = new MyMatrix(newMat);
                return new_M;
            }
            public MyMatrix Multilpy(MyMatrix M)
            {
                //矩阵乘法
                if(this.Cols!= M.Rows)
                {
                    MessageBox.Show("matrixs do not match");
                    return null;
                }
                double[,] newMat = new double[this.Rows, M.Cols];
                for(int i = 0; i < Rows; i++)
                {
                    for(int j = 0; j < M.Cols; j++)
                    {
                        for(int x = 0;x< this.Cols; x++)
                        {
                            newMat[i, j] += this.Value[i, x] * M.Value[x, j];
                        }
                    }
                }
                var new_m = new MyMatrix(newMat);
                return new_m;
            }
            public MyMatrix Inverse()
            {
                //矩阵求逆,利用伴随矩阵
                if (Rows != Cols)
                {
                    MessageBox.Show("not rectangular matrix!");
                    return null;
                }
                int dimension = Rows;
                double eps = 1e-6;
                double[,] W_mat = new double[dimension * 2, dimension * 2];
                
                for (int i = 0; i < dimension; i++)
                {
                    for (int j = 0; j < 2 * dimension; j++)
                    {
                        if (j < dimension)
                        {
                            W_mat[i, j] = Value[i, j];
                        }
                        else
                        {
                            W_mat[i, j] = j - dimension == i ? 1 : 0;
                        }
                    }
                }

                for (int i = 0; i < dimension; i++)
                {
                    if (Math.Abs(W_mat[i, i]) < eps)
                    //找到有对角线元素的一列
                    {
                        int j;
                        for (j = i + 1; j < dimension; j++)
                        {
                            if (Math.Abs(W_mat[j, i]) > eps) break;
                        }
                        if (j == dimension) return null;
                        //矩阵不可逆
                        for (int r = i; r < 2 * dimension; r++)
                        {
                            W_mat[i, r] += W_mat[j, r];
                            //整列移动
                        }
                    }
                    double ep = W_mat[i, i];
                    //正规化
                    for (int r = i; r < 2 * dimension; r++)
                    {
                        W_mat[i, r] = W_mat[i, r] / ep;
                    }

                    for (int j = i + 1; j < dimension; j++)
                    {
                        double e = -1 * (W_mat[j, i] / W_mat[i, i]);
                        for (int r = i; r < 2 * dimension; r++)
                        {
                            W_mat[j, r] += e * W_mat[i, r];
                        }
                    }
                }

                for (int i = dimension - 1; i >= 0; i--)
                {
                    for (int j = i - 1; j >= 0; j--)
                    {
                        double e = -1 * (W_mat[j, i] / W_mat[i, i]);
                        for (int r = i; r < 2 * dimension; r++)
                        {
                            W_mat[j, r] += e * W_mat[i, r];
                        }
                    }
                }

                double[,] result = new double[dimension, dimension];
                for (int i = 0; i < dimension; i++)
                {
                    for (int r = dimension; r < 2 * dimension; r++)
                    {
                        result[i, r - dimension] = W_mat[i, r];
                    }
                }
                var newMat = new MyMatrix(result);                
                return newMat;
            }
        }
        public MainWindow()
        {
            InitializeComponent();
            SaveCounter = 0;
            InterpolateMethod = 0;
            Rotate = 0;
            Radius = 0;
            DistortionRadus = 0;
        }
        //Button Events
        private void Rotate_change_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                Rotate = Convert.ToDouble(TextBox_Rotate.Text);
                Radius = Convert.ToDouble(TextBox_Radius.Text);
            }
            catch (FormatException)
            {
                MessageBox.Show("输入不合法，请重新输入。");
            }
            ImageRotate();
            ShowImg.Source = CurrentImg;
        }
        private void Distortion_Change_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                DistortionRadus = Convert.ToDouble(TextBox_DistortionRadius.Text);
            }
            catch (FormatException)
            {
                MessageBox.Show("输入不合法，请重新输入。");
            }
            ImageDistortion();
            ShowImg.Source = CurrentImg;
        }
        private void Wrap_Click(object sender, RoutedEventArgs e)
        {
            if(ProcessPoints() == 1)
            {
                ImageWrap();
                ShowImg.Source = CurrentImg;
            }
        }        
        //transformations
        private void ImageRotate()
        {
            //旋转变换
            if(CurrentImg == null)
            {
                MessageBox.Show("图片不能为空，请先载入图片");
                return;
            }
            CurrentImg = OriginalImg;
            var CurrentBitmap = BitmapImageToBitmap(CurrentImg);
            var width = CurrentImg.PixelWidth;
            var height = CurrentImg.PixelHeight;
            int[] center = new int[2];
            
            //处理输入
            if(IsInteger(Center_X.Text) && IsInteger(Center_Y.Text))
            {
                center[0] = Convert.ToInt32(Center_X.Text);
                center[1] = Convert.ToInt32(Center_Y.Text);
                if(center[0]>width || center[1] > height)
                {
                    MessageBox.Show("输入旋转中心有误，请输入[0, " + width.ToString() + "] [0, " + height.ToString() + "] 之间的整数值。");
                    return;
                }
            }
            else
            {
                MessageBox.Show("输入旋转中心有误，请输入[0, " + width.ToString() + "] [0, " + height.ToString() + "] 之间的整数值。");
                return;
            }
            //开始计算和插值
            for(int i = 0; i< width; i++) 
            {
                for(int j = 0; j<height; j ++)
                {
                    double DeltaX = i - center[0];
                    double DeltaY = j - center[1];
                    var distance = Math.Sqrt(DeltaX * DeltaX + DeltaY * DeltaY);
                    if (Radius == 0)
                    {
                        return;
                    }
                    if (distance <= Radius)
                    {
                        //计算
                        double a = Rotate * (Radius - distance) / Radius * Math.PI / 180.0;
                        var x = (i - center[0]) * Math.Cos(a) - (j - center[1]) * Math.Sin(a) + center[0];
                        var y = (i - center[0]) * Math.Sin(a) + (j - center[1]) * Math.Cos(a) + center[1];
                        //边界处理
                        if (x > width - 1 || x < 0 || y > height - 1 || y < 0)
                        {
                            CurrentBitmap.SetPixel(i, j, Color.FromArgb(0, 0, 0));
                            continue;
                        }
                        //插值
                        if (InterpolateMethod == 0)
                        {
                            CurrentBitmap.SetPixel(i, j, NearestInterpolation(x, y));
                        }
                        if (InterpolateMethod == 1)
                        {
                            CurrentBitmap.SetPixel(i, j, BilineaIntepolation(x, y, width, height));
                        }
                        if (InterpolateMethod == 2)
                        {
                            CurrentBitmap.SetPixel(i, j, BicubicInterpolation(x, y, width, height));
                        }
                    }
                }
            }
            CurrentImg = BitmapToBitmapImage(CurrentBitmap);            
        }
        private void ImageDistortion()
        {
            //图像畸变
            if(CurrentImg == null)
            {
                MessageBox.Show("图片不能为空，请先载入图片");
                return;
            }

            CurrentImg = OriginalImg;
            var CurrentBitmap = BitmapImageToBitmap(CurrentImg);
            var width = CurrentImg.PixelWidth;
            var height = CurrentImg.PixelHeight;
            int[] center = new int[2];
            //处理输入
            if (IsInteger(Center_X.Text) && IsInteger(Center_Y.Text))
            {

                center[0] = Convert.ToInt32(Center_X.Text);
                center[1] = Convert.ToInt32(Center_Y.Text);
                if (center[0] > width || center[1] > height)
                {
                    MessageBox.Show("输入旋转中心有误，请输入[0, " + width.ToString() + "] [0, " + height.ToString() + "] 之间的整数值。");
                    return;
                }
            }
            else
            {
                MessageBox.Show("输入旋转中心有误，请输入[0, " + width.ToString() + "] [0, " + height.ToString() + "] 之间的整数值。");
                return;
            }            
            //开始计算和插值
            for(int i = 0; i < width; i++)
            {
                for(int j = 0; j < height; j++)
                {
                    //计算
                    double DeltaX = i - center[0];
                    double DeltaY = j - center[1];
                    var distance = Math.Sqrt(DeltaX * DeltaX + DeltaY * DeltaY);
                    
                    var distance_pow2 = distance * distance;
                    var k = DistortionRadus * distance_pow2 / (width*height);
                    double x = i;
                    double y = j;
                    
                    x = (1 + k) * (i - center[0]) + center[0];
                    y = (1 + k) * (j - center[1]) + center[1];
                    //边界处理
                    if (x > width - 1 || x < 0 || y > height-1 || y < 0)
                    {
                        CurrentBitmap.SetPixel(i, j, Color.FromArgb(0, 0, 0));
                        continue;
                    }
                    //插值
                    if (InterpolateMethod == 0)
                    {
                        CurrentBitmap.SetPixel(i, j, NearestInterpolation(x, y));
                    }
                    if (InterpolateMethod == 1)
                    {
                        CurrentBitmap.SetPixel(i, j, BilineaIntepolation(x, y, width, height));
                    }
                    if (InterpolateMethod == 2)
                    {
                        CurrentBitmap.SetPixel(i, j, BicubicInterpolation(x, y, width, height));
                    }
                }
            }
            CurrentImg = BitmapToBitmapImage(CurrentBitmap);
        }
        private void ImageWrap()
        {
            //TPS变换
            CurrentImg = OriginalImg;
            var CurrentBitmap = BitmapImageToBitmap(CurrentImg);
            var width = CurrentImg.PixelWidth;
            var height = CurrentImg.PixelHeight;

            

            //Affine
            MyMatrix MControlPoints = new MyMatrix(OriginalFacePoints);
            double[,] qq = new double[68, 1];
            for (int ii = 0; ii < 68; ii++)
                qq[ii, 0] = 1;
            MyMatrix one_matrix = new MyMatrix(qq);

            MControlPoints.Append(one_matrix, 0);
            MyMatrix V = new MyMatrix(ChangeFacePoints);
            MControlPoints = MControlPoints.Transpose().Multilpy(MControlPoints).Inverse().Multilpy(MControlPoints.Transpose()).Multilpy(V);
            for (int i = 0; i < 68; i++)
            {
                OriginalFacePoints[i, 0] = MControlPoints.Value[0,0]* OriginalFacePoints[i, 0] + MControlPoints.Value[1, 0] * OriginalFacePoints[i, 1] + MControlPoints.Value[2, 0];
                OriginalFacePoints[i, 1] = MControlPoints.Value[0, 1] * OriginalFacePoints[i, 0] + MControlPoints.Value[1, 1] * OriginalFacePoints[i, 1] + MControlPoints.Value[2, 1];
            }

            //P矩阵
            double[,] q = new double[68, 1];
            for (int ii = 0; ii < 68; ii++)
                q[ii, 0] = 1;
            MyMatrix P_matrix = new MyMatrix(q);
            MyMatrix Ori = new MyMatrix(OriginalFacePoints);
            P_matrix.Append(Ori, 0);

            //K矩阵
            double[,] K = new double[68, 68];

            for(int i = 0; i < 68; i++)
            {
                for(int j = 0; j < 68; j++)
                {
                    double dis = Math.Sqrt(Math.Pow((OriginalFacePoints[i, 0] - OriginalFacePoints[j, 0]),2)
                       + Math.Pow((OriginalFacePoints[i, 1] - OriginalFacePoints[j, 1]), 2));
                    K[i, j] = _U_fun(dis);
                }
            }
            //L矩阵
            MyMatrix K_matrix = new MyMatrix(K);
            MyMatrix L_matrix = new MyMatrix(K_matrix);
            L_matrix.Append(P_matrix, 0);
            double[,] Zeros33 = new double[3, 3];
            MyMatrix Z = new MyMatrix(Zeros33);
            var P_t = P_matrix.Transpose();
            P_t.Append(Z, 0);
            L_matrix.Append(P_t, 1);
            //V矩阵
            MyMatrix V_matrix = new MyMatrix(ChangeFacePoints);
            double[,] zeros23 = new double[2, 3];
            MyMatrix zer23 = new MyMatrix(zeros23);
            var Y_matrix = new MyMatrix(V_matrix);
            //Y矩阵
            Y_matrix = Y_matrix.Transpose();
            Y_matrix.Append(zer23, 0);
            Y_matrix = Y_matrix.Transpose();
            //通过求逆,相乘的方法求解线性方程组 L[w_1, ..., w_n, a_1, a_x, a_y]^T = Y
            var L_inverted = L_matrix.Inverse();
            var L_inv_Y = L_inverted.Multilpy(Y_matrix);
            //结果
            var weight = L_inv_Y.Cut(68, 2);
            var a1 = new double[2];
            a1[0] = L_inv_Y.Value[68, 0];
            a1[1] = L_inv_Y.Value[68, 1];
            var aX = new double[2];
            aX[0] = L_inv_Y.Value[69, 0];
            aX[1] = L_inv_Y.Value[69, 1];
            var aY = new double[2];
            aY[0] = L_inv_Y.Value[70, 0];
            aY[1] = L_inv_Y.Value[70, 1];
            //开始计算
            for (int i = 0; i < width; i++)
            {
                for(int j = 0; j < height; j++)
                {
                    double[] sum = new double[2];
                    double distance = 0;
                    for(int m = 0; m < 68; m++)
                    {
                        distance = Math.Sqrt(Math.Pow(OriginalFacePoints[m,0]-i, 2)
                            + Math.Pow(OriginalFacePoints[m, 1]-j, 2));
                        //distance = OriginalFacePoints[m, 0] - i + OriginalFacePoints[m, 1] - j;
                        sum[0] += weight.Value[m, 0] * _U_fun(distance);
                        sum[1] += weight.Value[m, 1] * _U_fun(distance);
                    }
                    //计算对应点
                    var x = a1[0] + aX[0] * (i) + aY[0] * (j) + sum[0];
                    var y = a1[1] + aX[1] * (i) + aY[1] * (j) + sum[1];
                    if (x > width - 1 || x < 0 || y > height - 1 || y < 0)
                    {
                        CurrentBitmap.SetPixel(i, j, Color.FromArgb(0, 0, 0));
                        continue;
                    }
                    //插值
                    if (InterpolateMethod == 0)
                    {
                        CurrentBitmap.SetPixel(i, j, NearestInterpolation(x, y));
                    }
                    if (InterpolateMethod == 1)
                    {
                        CurrentBitmap.SetPixel(i, j, BilineaIntepolation(x, y, width, height));
                    }
                    if (InterpolateMethod == 2)
                    {
                        CurrentBitmap.SetPixel(i, j, BicubicInterpolation(x, y, width, height));
                    }
                }
            }
            CurrentImg = BitmapToBitmapImage(CurrentBitmap);
        }
        //interpolations
        private Color NearestInterpolation(double x, double y)
        {
            //最近邻插值
            int nx = (int)Math.Round(x);
            int ny = (int)Math.Round(y);
            return Color.FromArgb(OriginalMatrix[nx, ny, 0], OriginalMatrix[nx, ny, 1], OriginalMatrix[nx, ny, 2]);
        }
        private Color BilineaIntepolation(double x, double y, int width, int height)
        {
            //双线性插值
            int a = (int)Math.Floor(x);
            int b = (int)Math.Floor(y);
            int c = a + 1;
            int d = b + 1;
            //边界处理
            if(c>= width)            
                c = width-1;            
            if(d>= height)            
                d = height-1;

            double u = x - a;
            double v = y - b;

            double[] inter = new double[3];
            //插值
            for(int i = 0; i < 3; i++)
            {
                inter[i] = u * v * OriginalMatrix[a, b, i] +
                   u * (1 - v) * OriginalMatrix[c, b, i] +
                   (1 - u) * (v) * OriginalMatrix[a, d, i] +
                   (1 - u) * (1 - v) * OriginalMatrix[c, d, i];
            }

            byte rr = (byte)Math.Round(inter[0]);
            byte gg = (byte)Math.Round(inter[1]);
            byte bb = (byte)Math.Round(inter[2]);

            return Color.FromArgb(rr, gg, bb);
        }
        private Color BicubicInterpolation(double x, double y, int width, int height)
        {
            //双三次插值
            var i = (int)Math.Floor(x);
            var j = (int)Math.Floor(y);
            double u = x - i;
            double v = y - j;
            double[] A = new double[4];
            A[0] = _S_fun(u + 1);
            A[1] = _S_fun(u);
            A[2] = _S_fun(u - 1);
            A[3] = _S_fun(u - 2);

            double[] C = new double[4];
            C[0] = _S_fun(v + 1);
            C[1] = _S_fun(v);
            C[2] = _S_fun(v - 1);
            C[3] = _S_fun(v - 2);
            double[][][] B = new double[3][][];
            for (int c = 0; c < 3; c++)
            {
                B[c] = new double[4][];
                for (int jj = 0; jj < 4; jj++)
                {
                    B[c][jj] = new double[4];
                    for (int ii = 0; ii < 4; ii++)
                    {
                        var x0 = i - 1 + ii;
                        var y0 = j - 1 + jj;
                        if(x0 < 0)
                        {
                            x0 = 0;
                        }
                        if (x0 >= width)
                        {
                            x0 = width - 1;
                        }
                        if (y0 < 0)
                        {
                            y0 = 0;
                        }
                        if (y0 >= height)
                        {
                            y0 = height - 1;
                        }
                        B[c][jj][ii] = OriginalMatrix[x0, y0, c];
                    }
                }
            }

            var inter = new double[3];
            for (int c = 0; c < 3; c++)
            {
                var AB = new double[4];
                for(int ii = 0; ii < 4; ii++)
                {

                    for(int jj = 0; jj < 4; jj++)
                    {
                        AB[ii] += A[jj] * B[c][ii][jj];
                    }

                }
                for(int ii = 0; ii < 4; ii++)
                {
                    inter[c] += AB[ii] * C[ii];
                }
                if (inter[c] <= 0)
                {
                    inter[c] = 0;
                }
                if(inter[c] >= 255)
                {
                    inter[c] = 255;
                }
            }
            

            byte rr = (byte)Math.Round(inter[0]);
            byte gg = (byte)Math.Round(inter[1]);
            byte bb = (byte)Math.Round(inter[2]);

            return Color.FromArgb(rr, gg, bb);
        }
        private double _S_fun(double x)
        {
            //bicubic中的函数
            double x_abs = Math.Abs(x);
            double r = 0;
            if(x_abs <= 1)
            {
                r = 1 - 2 * x_abs * x_abs + x_abs * x_abs * x_abs;
            }
            else if(x_abs < 2)
            {
                r = 4 - 8 * x_abs + 5 * x_abs * x_abs - x_abs * x_abs * x_abs;
            }            
            return r;
        }
        private double _U_fun(double r)
        {
            //TPS中的U函数
            if (r == 0)
            {
                return 0;
            }
            else
            {
                var r_2 = r * r;
                return r_2 * Math.Log(r_2);
            }
        }
        //process img functions
        private void SaveImg_Click(object sender, RoutedEventArgs e)
        {
            if (CurrentImg != null)
            {
                SaveCounter++;
            }
            var o_path = ".\\ResultsData";
            if (Directory.Exists(o_path) == false)
            {
                Directory.CreateDirectory(o_path);
            }
            var filePath = ".\\ResultsData\\Result" + SaveCounter.ToString() + ".png";
            while (true)
            {
                if(Directory.Exists(filePath) == true)
                {
                    SaveCounter++;
                    filePath = ".\\ResultsData\\Result" + SaveCounter.ToString() + ".png";
                    continue;
                }
                break;
            }
            if (SaveBitmapImageIntoFile(CurrentImg, filePath) == 1)
            {
                MessageBox.Show("结果已保存为" + filePath);
            }            
        }
        private void Restore_Click(object sender, RoutedEventArgs e)
        {
            //恢复原图
            if (CurrentImg == null)
            {
                MessageBox.Show("图片不能为空，请先载入图片");
                return;
            }
            CurrentImg = OriginalImg;
            ShowImg.Source = CurrentImg;
        }
        private void ReadImg_Click(object sender, RoutedEventArgs e)
        {
            //读取图片
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Title = "选择文件";
            openFileDialog.Filter = "jpg|*.jpg|jpeg|*.jpeg";
            openFileDialog.FileName = string.Empty;
            openFileDialog.FilterIndex = 1;
            openFileDialog.RestoreDirectory = true;
            openFileDialog.DefaultExt = "jpg";
            bool? result = openFileDialog.ShowDialog();
            if (result != true)
            {
                return;
            }
            string fileName = openFileDialog.FileName;
            OriginalImgPath = fileName.Split('.')[0];
            BitmapImage bi = new BitmapImage();
            bi.BeginInit();
            bi.UriSource = new Uri(fileName);
            bi.EndInit();
            OriginalImg = bi;
            CurrentImg = bi;
            var b = BitmapImageToBitmap(bi);
            GenerateMatrix(b, 1);
            ShowImg.Source = OriginalImg;
            Center_X.Text = Math.Floor((double)(OriginalImg.PixelWidth/2)).ToString();
            Center_Y.Text = Math.Floor((double)(OriginalImg.PixelHeight/2)).ToString();
        }
        private void ReadImg_Face_Click(object sender, RoutedEventArgs e)
        {
            // Read FaceImg and .txt file
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Title = "选择文件";
            openFileDialog.Filter = "jpg|*.jpg|jpeg|*.jpeg";
            openFileDialog.FileName = string.Empty;
            openFileDialog.FilterIndex = 1;
            openFileDialog.RestoreDirectory = true;
            openFileDialog.DefaultExt = "jpg";
            bool? result = openFileDialog.ShowDialog();
            if (result != true)
            {
                return;
            }
            string fileName = openFileDialog.FileName;
            FaceImgPath = fileName.Split('.')[0];
            BitmapImage bi = new BitmapImage();
            bi.BeginInit();
            bi.UriSource = new Uri(fileName);
            bi.EndInit();
            FaceImg = bi;
            var b = BitmapImageToBitmap(bi);
            GenerateMatrix(b, 0);
            ShowImg_Face.Source = FaceImg;            
        }
        private int ProcessPoints()
        {
            // Process CriticalPoints
            ChangeFacePoints = null;
            ChangeFacePoints = new double[68, 2];
            StreamReader file;
            int counter = 0;
            try
            {
                file = new StreamReader(@OriginalImgPath + ".txt");
            }
            catch (FileNotFoundException)
            {
                MessageBox.Show("未找到原图对应关键点文件。请重新导入。");
                return 0;
            }
            string line;
            while ((line = file.ReadLine()) != null)
            {
                string[] ab = line.Split(' ');
                ChangeFacePoints[counter, 0] = Convert.ToInt32(Convert.ToDouble(ab[0]));
                ChangeFacePoints[counter, 1] = Convert.ToInt32(Convert.ToDouble(ab[1]));
                counter++;
            }
            OriginalFacePoints = null;
            OriginalFacePoints = new double[68, 2];
            counter = 0;
            StreamReader file1;
            try
            {
                file1 = new StreamReader(@FaceImgPath + ".txt");
            }
            catch (FileNotFoundException)
            {
                MessageBox.Show("未找到变换图对应关键点文件。请重新导入。");
                return 0 ;
            }
            string line1;
            while ((line1 = file1.ReadLine()) != null)
            {
                string[] ab = line1.Split(' ');

                OriginalFacePoints[counter, 0] = Convert.ToInt32(Convert.ToDouble(ab[0]));
                OriginalFacePoints[counter, 1] = Convert.ToInt32(Convert.ToDouble(ab[1]));
                counter++;
            }
            return 1;
        }
        private int SaveBitmapImageIntoFile(BitmapImage bitmapImage, string filePath)
        {
            //保存图片
            if(bitmapImage == null)
            {
                MessageBox.Show("保存图片不能为空！请先载入图片");
                return 0;
            }
            BitmapEncoder encoder = new PngBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(bitmapImage));
            
            using (var fileStream = new FileStream(filePath, FileMode.Create))
            {
                encoder.Save(fileStream);
            }
            return 1;
        }
        private void Bilinear_Unchecked(object sender, RoutedEventArgs e)
        {            
            if (Nearest.IsChecked == true)            
                InterpolateMethod = 0;            
            if (Bilinear.IsChecked == true)            
                InterpolateMethod = 1;            
            if (Bicubic.IsChecked == true)            
                InterpolateMethod = 2;            
        }
        private void Bicubic_Unchecked(object sender, RoutedEventArgs e)
        {
            if (Nearest.IsChecked == true)            
                InterpolateMethod = 0;            
            if (Bilinear.IsChecked == true)            
                InterpolateMethod = 1;            
            if (Bicubic.IsChecked == true)            
                InterpolateMethod = 2;            
        }
        private void Nearest_Unchecked(object sender, RoutedEventArgs e)
        {
            if (Nearest.IsChecked == true)            
                InterpolateMethod = 0;            
            if (Bilinear.IsChecked == true)            
                InterpolateMethod = 1;            
            if (Bicubic.IsChecked == true)            
                InterpolateMethod = 2;            
        }
        private void GenerateMatrix(Bitmap b, int s)
        {            
            if (s == 1)
            {
                //Generate OriginalMatrix
                OriginalMatrix = null;
                OriginalMatrix = new byte[(int)b.Width, (int)b.Height, 3];
                for (int i = 0; i < (int)b.Width; i++)
                {
                    for (int j = 0; j < (int)b.Height; j++)
                    {
                        OriginalMatrix[i, j, 0] = b.GetPixel(i, j).R;
                        OriginalMatrix[i, j, 1] = b.GetPixel(i, j).G;
                        OriginalMatrix[i, j, 2] = b.GetPixel(i, j).B;
                    }
                }
            }
            else
            {
                //Generate FaceMatrix
                FaceMatrix = null;
                FaceMatrix = new byte[(int)b.Width, (int)b.Height, 3];
                for (int i = 0; i < (int)b.Width; i++)
                {
                    for (int j = 0; j < (int)b.Height; j++)
                    {
                        FaceMatrix[i, j, 0] = b.GetPixel(i, j).R;
                        FaceMatrix[i, j, 1] = b.GetPixel(i, j).G;
                        FaceMatrix[i, j, 2] = b.GetPixel(i, j).B;
                    }
                }
            }            
        }
        private Bitmap BitmapImageToBitmap(BitmapImage bitmapImage)
        {
            //BitmapImage到Bitmap转换
            using (MemoryStream outStream = new MemoryStream())
            {
                BitmapEncoder enc = new BmpBitmapEncoder();
                enc.Frames.Add(BitmapFrame.Create(bitmapImage));
                enc.Save(outStream);
                Bitmap bitmap = new Bitmap(outStream);
                return new Bitmap(bitmap);
            }
        }
        private BitmapImage BitmapToBitmapImage(Bitmap bitmap)
        {
            //Bitmap向BitmapImage转换
            BitmapImage bitmapImage = new BitmapImage();
            MemoryStream ms = new MemoryStream();            
            bitmap.Save(ms, System.Drawing.Imaging.ImageFormat.Bmp);
            bitmapImage.BeginInit();
            bitmapImage.StreamSource = ms;
            bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
            bitmapImage.EndInit();
            bitmapImage.Freeze();            
            return bitmapImage;
        }
        // normal functions
        private bool IsInteger(string value)
        {
            string pattern = @"^[0-9]*[0-9][0-9]*$";
            return Regex.IsMatch(value, pattern);
        }
    }
}
