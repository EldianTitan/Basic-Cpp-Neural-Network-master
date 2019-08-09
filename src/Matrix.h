#pragma once

#include <memory>
#include <functional>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

enum MatrixAxis
{
	AXIS_HORIZONTAL,
	AXIS_VERTICAL
};

class Matrix
{
private:
	unsigned int m_rows = 0;
	unsigned int m_columns = 0;

	std::shared_ptr<float> m_data = nullptr;
private:
	void copy(const Matrix& other)
	{
		m_rows = other.m_rows;
		m_columns = other.m_columns;
		m_data = other.m_data;
	}
public:
	Matrix(int rows, int columns) :
		m_rows(rows), m_columns(columns)
	{
		m_data = std::shared_ptr<float>(new float[m_rows * m_columns], std::default_delete<float[]>());
	}

	Matrix(const Matrix& other) { copy(other); }

	inline Matrix& initValue(float value)
	{
		for(unsigned int i = 0; i < m_rows * m_columns; ++i)
		{
			m_data.get()[i] = value;
		}

		return *this;
	}

	inline Matrix& initValues(float* values)
	{
		for(unsigned int i = 0; i < m_rows * m_columns; ++i)
		{
			m_data.get()[i] = values[i];
		}

		return *this;
	}
	
	inline Matrix& initValues(std::initializer_list<float> list)
	{
		size_t i = 0;
		for(const float* it = list.begin(); it < list.end(); ++it)
		{
			if(i >= list.size() || i >= m_rows * m_columns) continue;

			m_data.get()[i] = *it;
			++i;
		}

		return *this;
	}

	inline Matrix& apply(std::function<float(float)> modifier)
	{
		return apply([&](float value, int row, int column, int numRows, int numColumns) -> float {
			return modifier(value);
		});

		std::cout << std::endl;
	}

	/**
	 * std::function<float(float value, int row, int column, int numRows, int numColumns)>
	*/
	inline Matrix& apply(std::function<float(float, int, int, int, int)> modifier)
	{
		for(unsigned int i = 0; i < m_rows; ++i)
		{
			for(unsigned int j = 0; j < m_columns; ++j)
			{
				m_data.get()[i * m_columns + j] = modifier(m_data.get()[i * m_columns + j], i, j, m_rows, m_columns);
			}
		}
		
		return *this;
	}

	inline Matrix applyCopy(std::function<float(float)> modifier) const
	{
		return applyCopy([&](float value, int row, int column, int numRows, int numColumns) -> float {
			return modifier(value);
		});
	}

	/**
	 * std::function<float(float value, int row, int column, int numRows, int numColumns)>
	*/
	inline Matrix applyCopy(std::function<float(float, int, int, int, int)> modifier) const
	{
		Matrix mat(m_rows, m_columns);

		for(unsigned int i = 0; i < m_rows; ++i)
		{
			for(unsigned int j = 0; j < m_columns; ++j)
			{
				mat.setValue(i, j, modifier(getValue(i, j), i, j, m_rows, m_columns));
			}
		}
		
		return mat;
	}

	inline Matrix dot(const Matrix& right) const
	{
		assert(m_columns == right.m_rows);

		Matrix mat(m_rows, right.m_columns);

		for(unsigned int i = 0; i < m_rows; ++i)
		{
			for(unsigned int j = 0; j < right.m_columns; ++j)
			{
				float value = 0;

				for(unsigned int k = 0; k < m_columns; ++k)
				{
					value += getValue(i, k) * right.getValue(k, j);
				}

				mat.setValue(i, j, value);
			}
		}

		return mat;
	}

	inline Matrix transpose() const
	{
		Matrix mat(m_columns, m_rows);

		for(unsigned int i = 0; i < m_rows; ++i)
		{
			for(unsigned int j = 0; j < m_columns; ++j)
			{
				mat.setValue(j, i, getValue(i, j));
			}
		}

		return mat;
	}

	inline Matrix sumAcross(MatrixAxis axis)
	{
		return Matrix::sumAcross(*this, axis);
	}

	inline Matrix operator+(float right) const { return applyCopy([=](float value) -> float { return value + right; }); }
	inline Matrix operator-(float right) const { return applyCopy([=](float value) -> float { return value - right; }); }
	inline Matrix operator*(float right) const { return applyCopy([=](float value) -> float { return value * right; }); }
	inline Matrix operator/(float right) const { return applyCopy([=](float value) -> float { return value / right; }); }

	inline Matrix operator+(const Matrix& right)
	{
		Matrix mat(std::max(m_rows, right.m_rows), std::max(m_columns, right.m_columns));
		for(unsigned int i = 0; i < mat.getRows(); ++i)
		{
			for(unsigned int j = 0; j < mat.getColumns(); ++j)
			{
				mat.setValue(i, j, getValue(i % m_rows, j % m_columns) + right.getValue(i % right.m_rows, j % right.m_columns));
			}
		}

		return mat;
	}

	inline Matrix operator-(const Matrix& right)
	{
		Matrix mat(std::max(m_rows, right.m_rows), std::max(m_columns, right.m_columns));
		for(unsigned int i = 0; i < mat.getRows(); ++i)
		{
			for(unsigned int j = 0; j < mat.getColumns(); ++j)
			{
				mat.setValue(i, j, getValue(i % m_rows, j % m_columns) - right.getValue(i % right.m_rows, j % right.m_columns));
			}
		}

		return mat;
	}

	inline Matrix operator*(const Matrix& right)
	{
		Matrix mat(std::max(m_rows, right.m_rows), std::max(m_columns, right.m_columns));
		for(unsigned int i = 0; i < mat.getRows(); ++i)
		{
			for(unsigned int j = 0; j < mat.getColumns(); ++j)
			{
				mat.setValue(i, j, getValue(i % m_rows, j % m_columns) * right.getValue(i % right.m_rows, j % right.m_columns));
			}
		}

		return mat;
	}

	inline Matrix operator/(const Matrix& right)
	{
		Matrix mat(std::max(m_rows, right.m_rows), std::max(m_columns, right.m_columns));
		for(unsigned int i = 0; i < mat.getRows(); ++i)
		{
			for(unsigned int j = 0; j < mat.getColumns(); ++j)
			{
				mat.setValue(i, j, getValue(i % m_rows, j % m_columns) / right.getValue(i % right.m_rows, j % right.m_columns));
			}
		}

		return mat;
	}

	inline Matrix operator-() { return applyCopy([](float value) -> float { return -value; }); }

	inline void operator=(const Matrix& other) { copy(other); }

	std::string toString(int precision = 2, const std::vector<std::string>& lineIndentations = {}) const;
	friend std::ostream& operator<<(std::ostream& stream, const Matrix& matrix);

	inline Matrix copy() const
	{
		Matrix mat(m_rows, m_columns);
		memcpy(mat.m_data.get(), m_data.get(), m_rows * m_columns * sizeof(float));

		return mat;
	}

	inline void setValue(unsigned int row, unsigned int column, float value) { m_data.get()[row * m_columns + column] = value; }
	inline float getValue(unsigned int row, unsigned int column) const { return m_data.get()[row * m_columns + column]; }

	inline unsigned int getRows() const { return m_rows; }
	inline unsigned int getColumns() const { return m_columns; }
public:
	inline static Matrix sumAcross(const Matrix& matrix, MatrixAxis axis)
	{
		if(axis == AXIS_HORIZONTAL)
		{
			Matrix mat(matrix.getRows(), 1);

			for(unsigned int i = 0; i < matrix.getRows(); ++i)
			{
				float value = 0;
				for(unsigned int j = 0; j < matrix.getColumns(); ++j)
				{
					value += matrix.getValue(i, j);
				}

				mat.setValue(i, 0, value);
			}

			return mat;
		}
		else
		{
			Matrix mat(1, matrix.getColumns());

			for(unsigned int i = 0; i < matrix.getColumns(); ++i)
			{
				float value = 0;
				for(unsigned int j = 0; j < matrix.getRows(); ++j)
				{
					value += matrix.getValue(j, i);
				}

				mat.setValue(0, i, value);
			}

			return mat;
		}
	}
};

inline Matrix operator+(float left, Matrix& mat)
{
	return mat.applyCopy([=](float value) -> float { return left + value; });
}

inline Matrix operator-(float left, Matrix& mat)
{
	return mat.applyCopy([=](float value) -> float { return left - value; });
}

inline Matrix operator*(float left, Matrix& mat)
{
	return mat.applyCopy([=](float value) -> float { return left * value; });
}

inline Matrix operator/(float left, Matrix& mat)
{
	return mat.applyCopy([=](float value) -> float { return left / value; });
}