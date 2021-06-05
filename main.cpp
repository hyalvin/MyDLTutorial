//===================================
//main.cpp
//C++学习中的练习代码_钱能
//===================================
#include<iostream>
#include<stdio.h>
#include<string.h>
#include<string>
#include<fstream>
#include<math.h>
#include<vector>
using namespace std;
//-----------------------------------

//1-1：输出一句话
/*
int main()
{
	std::cout << "I am a student." << endl;
}
*/

//1-2：输出自定义阶数的菱形
/*
int main()
{
	cout<<"请输入菱形阶数:";
	int n;
	cin >> n;//输入流赋值语句
	printf("\n");
	char array[100][100];
	if (n > 100 || n <= 1 || n % 2 == 0)
		printf("input error");
	else
	{
		int i, j;
		for (i = 0; i < (n + 1) / 2; i++)
		{
			for (j = 0; j < n; j++)
			{
				if ((n + 1) / 2 - i - 1 <= j && j<= (n + 1) / 2 + i - 1)
				{
					array[i][j] = '*';
				}
				else
				{
					array[i][j] = ' ';
				}
			}
		}
		for (i = (n+1)/2; i < n ; i++)
		{
			for (j = 0; j < n; j++)
			{
				(-(n + 1) / 2 + i + 1 <= j && j<= (n + 1) / 2 + n - i - 2) ? \
					array[i][j] = '*' : array[i][j] = ' ';
				//三目运算符a?x:y
				//续行符用\且后面不能再有无关字符
			}
		}
		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n; j++)
			{
				printf("%c", array[i][j]);
			}
			printf("\n\n");
		}
	}
}
*/

//输出自定义菱形改版,采用输出流方法
/*
int main()
{
	cout<<"请输入菱形阶数:";
	int n;
	cin >> n;//输入流赋值语句
	printf("\n");
	char array[100][100];
	if (n > 100 || n <= 1 || n % 2 == 0)
		printf("input error");
	else
	{
		for (int i = 1; i <= (n + 1) / 2; i++)
		{
			cout << string((n + 1) / 2 - i, ' ') + string(2 * i - 1, '*') + "\n" << endl;
		}
		for (int i = (n + 1) / 2 + 1; i <= n; i++)
		{
			cout << string(i - (n + 1) / 2, ' ') + string(2 * n - 2 * i + 1, '*') + "\n" << endl;
		}
	}
	
}
*/

//文件流练习：复制文本
/*
int main()
{
	ifstream in("Readme.txt");
	ofstream out("Readme_Copy.txt");
	for (string str; getline(in, str);)//getline函数一定要include<string>
		out << str << endl;
}
*/

//逻辑判断问题：百钱买百鸡。
//最简单直接的方法是多重循环以及continue语句逐条件判断，只要有一个条件不成立就退出当前循环
//当然也可以用if语句判断，若条件成立则打印方案
/*
int main()
{
	int money = 100, number = 100;//100块钱，100只鸡
	int cock, hen, chick;//公鸡，母鸡，小鸡
	int i=0;//记录方案数
	//这里由于三种鸡的数目不独立，只需做两次循环即可，这也是代码优化
	for (cock = 1; cock <= 13; cock++)
	{
		for (hen = 1; hen <= 18; hen++)
		{
			if (cock * 7 + hen * 5 + (100 - cock - hen) / 3 != money)
				continue;
			if ((100 - cock - hen) % 3 != 0)
				continue;
			i++;
			std::cout << "方案" << i << ": ";
			std::cout << "公鸡买" << cock << "只，" << "母鸡买" << hen << "只，" << "小鸡买" << 100 - cock - hen << "只。" << endl;
		}
	}
}
*/

//级数逼近
/*
int main()
{
	double sum = 0;
	for (double i = 1;; i++)
	{
		sum += pow(-1, i - 1) / (2 * i - 1);
		if (1 / (2 * i + 1) < pow(10, -6))
		{
			sum += pow(-1, i) / (2 * i + 1);
			break;
		}
	}
	cout << "精确到小数点后6位时，pi的近似值是" << fixed << sum * 4 << endl;
}
*/

//先用递归写一个阶乘函数，方便做级数逼近
/*
int factorial(int n)
{
	if (n == 1)
		return 1;
	else
		return n * factorial(n - 1);
}
*/
//2-1-1:for循环级数逼近
/*
int main()
{
	printf("请输入一个数：");
	double x, sum = 1;
	cin >> x;
	for (int i = 1;; i++)
	{
		if ( pow(x, i + 1) / factorial(i + 1) < 1e-8)
			break;
		sum += pow(-1, i + 1) * pow(x, i) / factorial(i);
	}
	std::cout.precision(8);
	std::cout << "当逼近精度为小数点后八位时，级数约为" <<sum << endl;
}
*/
//2-1-2:while循环级数逼近
/*
int main()
{
	printf("请输入一个数：");
	double x, sum = 1;
	cin >> x;
	int i = 1;
	while ( pow(x, i + 1) / factorial(i + 1) >= 1e-8)
	{
		sum += pow(-1, i + 1) * pow(x, i) / factorial(i);
		i++;
	}
	cout.precision(8);
	cout << "当逼近精度为小数点后八位时，级数约为" << sum << endl;
}
*/

//2-2:级数求和
/*
int main()
{
	int sum = 0;
	for (int i = 1; i <= 12; i++)
		sum += factorial(i);
	cout << sum << endl;
}
*/

//2-3:水仙花数
/*
int main()
{
	int a, b, c;
	for (int i = 100; i <= 999; i++)
	{
		a = i / 100;
		b = (i % 100) / 10;
		c = (i % 100) % 10;
		if (i == pow(a, 3) + pow(b, 3) + pow(c, 3))
			cout << i << endl;
	}
}
*/

//2-4:完数
/*
int main()
{
	for (int i = 2; i <= 1000; i++)
	{
		int sum = 0;
		for (int j = 1; j < i; j++)
		{
			if (i % j == 0)
				sum += j;
		}
		if (sum == i)
			cout << i << endl;
	}
}
*/

//2-5:3位对称素数
/*
int main()
{
	for (int i = 100; i < 999; i++)
	{
		int count = 0;
		for (int j = 2; j <= sqrt(i); j++)
		{
			if (i % j == 0)
			{
				count = 1;
				break;
			}

		}
		if (count == 0)
		{
			int a, c;
			a = i / 100;
			c = i % 100 % 10;
			if (a == c)
			{
				cout << i << endl;
			}
		}
	}
}
*/

//2-6:猴子吃桃
/*
int main()
{
	int num = 1;
	for (int i = 1; i <= 9; i++)
	{
		num = 2 * (num + 1);
	}
	cout << num << endl;
}
*/

//2-7:循环语句打印图案
/*
int main()
{
	int n;
	cin >> n;
	if (n <= 1 || n % 2 == 0)
		return 1;
	for (int i = 1; i < (n + 1) / 2; i++)
	{
		cout << string((n + 1) / 2 - i, ' ') + string(2 * i - 1, '%') + "\n";
	}
	for (int i = (n + 1) / 2; i <= n; i++)
	{
		cout << string(i - (n + 1) / 2, ' ') + string(2 * n + 1 - 2 * i, '%') + "\n";
	}
}
*/

//2-8:循环语句打印图案
/*
int main()
{
	int n;
	cin >> n;
	if (n <= 0)
		return 1;
	for (int i = 1; i <= n; i++)
	{
		cout << string(n - i, ' ') + string(i, '#') + string(5, ' ') + string(i, '$') + "\n";
	}
}
*/

//2-9:循环语句打印图案
/*
int main()
{
	int n;
	cin >> n;
	for (int i = 1; i <= n; i++)
	{
		cout << string(i - 1, ' ');
		for (int j = 1; j <= n - i; j++)
		{
			cout << "ST";
		}
		cout << "S" << endl;
	}
}
*/

//2-10:母牛生子
/*
int main()
{
	int n;
	cin >> n;
	if (n <= 3)
	{
		cout << 1 << endl;
		return 1;
	}


}
*/

//2-11:小球落地
/*
int main()
{
	double h = 100;
	double s = 0, h1;
	s += h;
	for (int i = 1; i < 10; i++)
	{
		h = h / 2;
		s += h * 2;
		if (i == 9)
		{
			h1 = h / 2;
			cout << s << "\n" << h1 << endl;
		}
	}
}
*/

//2-12:换钱
/*
int main()
{
	int ten, five, one;
	int count = 0;
	for (ten = 1; ten <= 9; ten++)
	{
		for (five = 1; five <= 17; five++)
		{
			if (100 - 10 * ten - 5 * five > 0)
			{
				count += 1;
				cout << "方案" << count << "兑10元" << ten << "张," << "兑5元" << five << "张," << "兑1元" << 100 - 10 * ten - 5 * five << "张." << endl;
			}
		}
	}
	cout << "\n" << "共有" << count << "种兑换方案" << endl;
}
*/

//2-13-1
/*
int main()
{
	string s = "0123456";
	for (int i = 1; i <= 6; i++)
	{
		cout << i << "   ";
		for (int j = i; j <= 7; j++)
		{
			cout << s[j - 1] << " ";
		}
		for (int j = 0; j <= i - 2; j++)
		{
			cout << s[j] << " ";
		}
		cout << "\n";
	}
}
*/

//2-13-2
/*
int main()
{
	for (int i = 1; i <= 6; i++)
	{
		for (int j = 1; j <= 7; j++)
		{
			cout << "(" << i << "," << j << ")" << " ";
		}
		cout << endl;
	}
}
*/

