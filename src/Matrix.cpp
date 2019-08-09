#include "Matrix.h"

#include <sstream>

std::string Matrix::toString(int precision, const std::vector<std::string>& lineIndentations) const
{
	using namespace std;

	const char tlCorner = (char)218;
	const char trCorner = (char)191;
	const char blCorner = (char)192;
	const char brCorner = (char)217;
	const char hLine = (char)179;
	const char vLine = (char)196;

	char buffer[64];

	stringstream output;
	string numberFormat;
	{
		stringstream ss; ss << precision;
		numberFormat = string("%.") + ss.str() + string("f");
	}

	int* lineLengths = new int[m_columns];
	memset(lineLengths, 0, m_columns * sizeof(int));

	for (unsigned int i = 0; i < m_rows; ++i)
	{
		int rowSize = 0;

		for (unsigned int j = 0; j < m_columns; ++j)
		{
			int length = sprintf_s(buffer, numberFormat.c_str(), getValue(i, j));
			lineLengths[j] = lineLengths[j] < length ? length : lineLengths[j];
		}
	}

	int maxRowSize = 0;
	for (unsigned int i = 0; i < m_columns; ++i)
	{
		maxRowSize += lineLengths[i];
	}

	unsigned int numAddSpaces = m_columns - 1;

	int numSpaces = maxRowSize + numAddSpaces + 1;
	char* spaces = new char[numSpaces];
	memset(spaces, ' ', sizeof(char) * (numSpaces - 1));
	spaces[numSpaces - 1] = 0;

	if(lineIndentations.size() > 0)
	{
		output << lineIndentations[0];
	}

	output << tlCorner << vLine << spaces << vLine << trCorner << "\n";
	for (unsigned int i = 0; i < m_rows; ++i)
	{
		int offset = 0;
		for (unsigned int j = 0; j < m_columns; ++j)
		{
			int n = sprintf_s(spaces + offset, numSpaces - offset, numberFormat.c_str(), getValue(i, j));

			spaces[offset + n] = ' ';
			if (j < numAddSpaces)
			{
				offset += lineLengths[j] + 1;
			}

			if (j >= numAddSpaces && offset + n == maxRowSize + numAddSpaces)
			{
				spaces[offset + n] = 0;
			}
		}

		if(lineIndentations.size() > 0)
		{
			output << lineIndentations[(i + 1) % lineIndentations.size()];
		}

		output << hLine << " " << spaces << " " << hLine << "\n";

		memset(spaces, ' ', sizeof(char) * (maxRowSize + numAddSpaces));
		spaces[maxRowSize + numAddSpaces] = 0;
	}

	if(lineIndentations.size() > 0)
	{
		output << lineIndentations[m_rows % lineIndentations.size()];
	}

	output << blCorner << vLine << spaces << vLine << brCorner;// << "\n";

	delete[] lineLengths;
	delete[] spaces;

	return output.str();
}

std::ostream& operator<<(std::ostream& stream, const Matrix& matrix)
{
	stream << matrix.toString();
	return stream;
}