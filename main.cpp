//===================================
//main.cpp
//C++ѧϰ�е���ϰ����_Ǯ��
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

//1-1�����һ�仰
/*
int main()
{
	std::cout << "I am a student." << endl;
}
*/

//1-2������Զ������������
/*
int main()
{
	cout<<"���������ν���:";
	int n;
	cin >> n;//��������ֵ���
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
				//��Ŀ�����a?x:y
				//���з���\�Һ��治�������޹��ַ�
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

//����Զ������θİ�,�������������
/*
int main()
{
	cout<<"���������ν���:";
	int n;
	cin >> n;//��������ֵ���
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

//�ļ�����ϰ�������ı�
/*
int main()
{
	ifstream in("Readme.txt");
	ofstream out("Readme_Copy.txt");
	for (string str; getline(in, str);)//getline����һ��Ҫinclude<string>
		out << str << endl;
}
*/

//�߼��ж����⣺��Ǯ��ټ���
//���ֱ�ӵķ����Ƕ���ѭ���Լ�continue����������жϣ�ֻҪ��һ���������������˳���ǰѭ��
//��ȻҲ������if����жϣ��������������ӡ����
/*
int main()
{
	int money = 100, number = 100;//100��Ǯ��100ֻ��
	int cock, hen, chick;//������ĸ����С��
	int i=0;//��¼������
	//�����������ּ�����Ŀ��������ֻ��������ѭ�����ɣ���Ҳ�Ǵ����Ż�
	for (cock = 1; cock <= 13; cock++)
	{
		for (hen = 1; hen <= 18; hen++)
		{
			if (cock * 7 + hen * 5 + (100 - cock - hen) / 3 != money)
				continue;
			if ((100 - cock - hen) % 3 != 0)
				continue;
			i++;
			std::cout << "����" << i << ": ";
			std::cout << "������" << cock << "ֻ��" << "ĸ����" << hen << "ֻ��" << "С����" << 100 - cock - hen << "ֻ��" << endl;
		}
	}
}
*/

//�����ƽ�
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
	cout << "��ȷ��С�����6λʱ��pi�Ľ���ֵ��" << fixed << sum * 4 << endl;
}
*/

//���õݹ�дһ���׳˺����������������ƽ�
/*
int factorial(int n)
{
	if (n == 1)
		return 1;
	else
		return n * factorial(n - 1);
}
*/
//2-1-1:forѭ�������ƽ�
/*
int main()
{
	printf("������һ������");
	double x, sum = 1;
	cin >> x;
	for (int i = 1;; i++)
	{
		if ( pow(x, i + 1) / factorial(i + 1) < 1e-8)
			break;
		sum += pow(-1, i + 1) * pow(x, i) / factorial(i);
	}
	std::cout.precision(8);
	std::cout << "���ƽ�����ΪС������λʱ������ԼΪ" <<sum << endl;
}
*/
//2-1-2:whileѭ�������ƽ�
/*
int main()
{
	printf("������һ������");
	double x, sum = 1;
	cin >> x;
	int i = 1;
	while ( pow(x, i + 1) / factorial(i + 1) >= 1e-8)
	{
		sum += pow(-1, i + 1) * pow(x, i) / factorial(i);
		i++;
	}
	cout.precision(8);
	cout << "���ƽ�����ΪС������λʱ������ԼΪ" << sum << endl;
}
*/

//2-2:�������
/*
int main()
{
	int sum = 0;
	for (int i = 1; i <= 12; i++)
		sum += factorial(i);
	cout << sum << endl;
}
*/

//2-3:ˮ�ɻ���
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

//2-4:����
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

//2-5:3λ�Գ�����
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

//2-6:���ӳ���
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

//2-7:ѭ������ӡͼ��
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

//2-8:ѭ������ӡͼ��
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

//2-9:ѭ������ӡͼ��
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

//2-10:ĸţ����
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

//2-11:С�����
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

//2-12:��Ǯ
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
				cout << "����" << count << "��10Ԫ" << ten << "��," << "��5Ԫ" << five << "��," << "��1Ԫ" << 100 - 10 * ten - 5 * five << "��." << endl;
			}
		}
	}
	cout << "\n" << "����" << count << "�ֶһ�����" << endl;
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

