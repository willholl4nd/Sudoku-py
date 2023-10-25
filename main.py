import pandas as pd
import numpy as np
import math

SQR_IDX_LOOKUP: dict[int, dict] = dict()

def setup_SQR_IDX_LOOKUP(width, height, sqr_width, sqr_height):
    global SQR_IDX_LOOKUP
    idx_table = []
    for i in range(height):
        idx_table.append(list())
        for j in range(width):
            idx_table[i].append(rc_to_idx(i,j,width))

    df = pd.DataFrame.from_records(idx_table)

    sqrs = list()
    for i in range(0, height, sqr_height):
        for j in range(0, width, sqr_width):
            i_end = i + sqr_height
            j_end = j + sqr_width

            index_pairs: list[tuple[int, int]] = list()
            for i_x in range(i, i_end):
                for j_x in range(j, j_end):
                    index_pairs.append((i_x, j_x))

            sqrs.append([(df.iloc[idx_pair], idx_to_rc(index, sqr_width, sqr_height)) for index, idx_pair in enumerate(index_pairs)])

    
    for i,sqr in enumerate(sqrs):
        for info in sqr:
            sqr_index = i
            sqr_local_rc = info[1]
            overall_index = info[0]
            SQR_IDX_LOOKUP[overall_index] = {"sqr_local_rc": sqr_local_rc, "sqr_index": sqr_index}

    print(SQR_IDX_LOOKUP)


def rc_to_sqr_idx(row, col, width) -> dict:
    global SQR_IDX_LOOKUP

    idx = rc_to_idx(row, col, width)
    return SQR_IDX_LOOKUP[idx]

def rc_to_idx(row, col, width) -> int:
    return int(row * width + col)

def idx_to_rc(idx, width, height) -> tuple[int, int]:
    row = math.floor(idx / width)
    col = idx - row * width
    return int(row), int(col)




class Cell:

    def __init__(self, initial_value, index=None, sqr_index=None, sqr_row=None, sqr_col=None): 
        self.__value = initial_value
        self.__index = index
        self.__sqr_index = sqr_index
        self.__sqr_row = sqr_row
        self.__sqr_col = sqr_col

    def get_value(self):
        return self.__value

    def set_value(self, value):
        old = self.__value
        self.__value = value
        return old

    def __repr__(self) -> str:
        return f'val:{self.__value},idx:{self.__index}'


class Row:

    def __init__(self, length): 
        self.__row = []
        self.__length = length

        # Stores the missing values in the row; keys are possibilities in a cell
        # and values are whether they are present or not
        self.__missing_values: dict[int, bool] = self.__setup_missing()

    def initialze_row(self, values) -> list:
        if len(values) != self.__length:
            raise Exception("Could not initialize the Column object because the number of values did not match the length of the column")

        old = self.__row.copy()
        for value in values:
            self.__row.append(value)

        return old

    def get_cells(self) -> list[Cell]:
        return self.__row

    def set_cell(self, index, cell_value: int) -> int:
        if index < 0 or index >= self.__length:
            raise IndexError("Cannot index outside Row cells")

        if cell_value < 1 or cell_value > 9:
            raise ValueError("Cell value cannot be set outside of the range 1 to 9 inclusive")

        old: int = self.__row[index].set_value(cell_value)
        self._update_missing(value=cell_value)

        return old

    def __setup_missing(self) -> dict[int, bool]:
        ret = dict()
        for i in range(self.__length):
            ret[i] = False

        return ret

    def _update_missing(self, value=None):
        # No value was given
        if not value:
            full = [i for i in range(9)] # The full list of possibilities
            add = [] # Values that need to be added to the missing list
            for cell in self.__row:
                # Cell value has actual value
                if (val := cell.get_value()) != 0:
                    add.append(val)

            remove = list(set(full) - set(add)) # List of values that need to be removed

            # Remove and add any missing values
            for val in remove:
                self.__missing_values[val] = False
            for val in add:
                self.__missing_values[val] = True

        else:
            self.__missing_values[value] = True



    def __repr__(self) -> str:
        return f'length: {self.__length}, values: {self.__row}'


class Column:

    def __init__(self, length): 
        self.__col = []
        self.__length = length
        self.__missing_values: dict[int, bool] = self.__setup_missing() # Stores the missing values in the column

    def initialze_col(self, values) -> list:
        if len(values) != self.__length:
            raise Exception("Could not initialize the Column object because the number of values did not match the length of the column")

        old = self.__col.copy()
        for value in values:
            self.__col.append(value)

        return old

    def get_cells(self) -> list[Cell]:
        return self.__col

    def set_cell(self, index, cell_value: int) -> int:
        if index < 0 or index >= self.__length:
            raise IndexError("Cannot index outside Column cells")

        if cell_value < 1 or cell_value > 9:
            raise ValueError("Cell value cannot be set outside of the range 1 to 9 inclusive")

        old: int = self.__col[index].set_value(cell_value)
        self._update_missing(value=cell_value)

        return old

    def __setup_missing(self) -> dict[int, bool]:
        ret = dict()
        for i in range(self.__length):
            ret[i] = False

        return ret

    def _update_missing(self, value=None):
            # No value was given
            if not value:
                full = [i for i in range(9)] # The full list of possibilities
                add = [] # Values that need to be added to the missing list
                for cell in self.__col:
                    # Cell value has actual value
                    if (val := cell.get_value()) != 0:
                        add.append(val)

                remove = list(set(full) - set(add)) # List of values that need to be removed

                # Remove and add any missing values
                for val in remove:
                    self.__missing_values[val] = False
                for val in add:
                    self.__missing_values[val] = True

            else:
                self.__missing_values[value] = True


    def __repr__(self) -> str:
        return f'length: {self.__length}, values: {self.__col}'


class Square:

    def __init__(self, width, height, col_length=3, row_length=3): 
        self.__cols = [Column(col_length) for _ in range(width)]
        self.__rows = [Row(row_length) for _ in range(height)]
        self.__width = width
        self.__height = height
        self.__sqr_width = col_length
        self.__sqr_height = row_length
        self.__missing_values: dict[int, bool] = self.__setup_missing() # Stores the missing values in the square


    def initialze_sqr(self, values: list[Cell]) -> None:
        '''
        Expect a one-dimensional list of items to be populated in a square 
        by index (e.g. index 0 goes in the first row and first column, index 1 
        goes in the first row and second column, etc.).

        This function is an in-place modification of the Square object.
        '''

        if len(values) != self.__width * self.__height:
            raise Exception("Could not initialize the Square object because the number of values did not match the dimensions of the square")
        
        cols = [[] for _ in range(self.__width)]
        rows = [[] for _ in range(self.__height)]

        # TODO Optimize since we are doing O(n + n^2) operations when we could do just O(n) operations
        for i,val in enumerate(values):
            col_index = i % self.__width
            cols[col_index].append(val)

        for i in range(self.__width):
            for j in range(self.__height):
                val = cols[j][i]
                rows[i].append(val)

        self.__init_cols(cols)
        self.__init_rows(rows)

            
    def __init_cols(self, columns: list[list[Cell]]) -> None:
        '''
        Initialize the __cols Columns with cells that are in the columns argument.

        This function is an in-place modification of the __cols instance variable.
        '''

        for i,col in enumerate(columns):
            self.__cols[i].initialze_col(col)


    def __init_rows(self, rows: list[list[Cell]]) -> None:
        '''
        Initialize the __rows Rows with cells that are in the rows argument.

        This function is an in-place modification of the __rows instance variable.
        '''

        for i,row in enumerate(rows):
            self.__rows[i].initialze_row(row)


    def __repr__(self) -> str:
        '''
        This generates a string representation of the Square object.
        '''

        row_string = '\n\t'.join([repr(r) for r in self.__rows])
        col_string = '\n\t'.join([repr(c) for c in self.__cols])
        return f'width: {self.__width}, height: {self.__height}\nrows: \n\t{row_string}\ncols: \n\t{col_string}'


    def get_row_cell(self, row_index, cell_index):
        '''
        Given an index (row_index) into the __rows instance list, we grab the 
        cell at the index (cell_index) in the Row object.
        '''

        if row_index < 0 or row_index >= self.__height:
            raise Exception("The row index you are trying to access is out of bounds")

        if cell_index < 0 or cell_index >= self.__width:
            raise Exception("The cell index you are trying to access is out of bounds")

        return self.__rows[row_index][cell_index]


    def get_col_cell(self, col_index, cell_index):
        '''
        Given an index (col_index) into the __cols instance list, we grab the 
        cell at the index (cell_index) in the Column object.
        '''

        if col_index < 0 or col_index >= self.__height:
            raise Exception("The column index you are trying to access is out of bounds")

        if cell_index < 0 or cell_index >= self.__width:
            raise Exception("The cell index you are trying to access is out of bounds")

        return self.__cols[col_index][cell_index]
    

    def get_cols(self) -> list[Column]:
        '''
        Returns the entire list of Column objects.
        '''

        return self.__cols


    def get_rows(self) -> list[Row]:
        '''
        Returns the entire list of Row objects.
        '''

        return self.__rows


    # TODO
    def set_cell(self, index, cell_value: int) -> int:
        if index < 0 or index >= self.__height:
            raise IndexError("Cannot index outside Square cells")

        if cell_value < 1 or cell_value > 9:
            raise ValueError("Cell value cannot be set outside of the range 1 to 9 inclusive")

        old: int = self.__col[index].set_value(cell_value)
        self._update_missing(value=cell_value)

        return old


    def __setup_missing(self) -> dict[int, bool]:
        '''
        Generates the __missing_values dictionary, where the key into the dict 
        is an int (the value of a valid cell), and the value is a bool (a present 
        indicator).

        This function is not in-place and returns the generated dictionary.
        '''

        ret = dict()
        for i in range(self.__sqr_height * self.__sqr_width):
            ret[i] = False

        return ret

    def _update_missing(self, value=None):
        '''
        This function takes an optional value argument. This function can have 
        two separate functionalities:
            1. A value not None means we have changed a zero to a valid entry 
            and we know exactly what possibility was taken away from the Square.
            2. A value None means that we don't know what value changed and we 
            need to do an entire recheck of the possibilities in the Square.

        This function is an in-place modification of the __missing_values 
        instance variable.
        '''

        # No value was given
        if not value:
            full = [i for i in range(9)] # The full list of possibilities
            add = [] # Values that need to be added to the missing list

            # Only need to check rows since cells are shared objects in the row and col lists
            for row in self.__rows:
                for cell in row.get_cells():
                    # Cell value has actual value
                    if (val := cell.get_value()) != 0:
                        add.append(val)

            remove = list(set(full) - set(add)) # List of values that need to be removed

            # Remove and add any missing values
            for val in remove:
                self.__missing_values[val] = False
            for val in add:
                self.__missing_values[val] = True

        # Value was given
        else:
            self.__missing_values[value] = True


class EntireSudoku:
    '''
    Contains the entire Sudoku problem inside various containers that facilitate
    solving the problem (rows, columns, squares).
    '''

    def __init__(self, dataframe: pd.DataFrame): 
        self.__width = 9
        self.__height = 9
        self.__sqr_width = int(temp_width) if (temp_width := math.sqrt(self.__width)) == int(temp_width) else 0
        self.__sqr_height = int(temp_height) if (temp_height := math.sqrt(self.__height)) == int(temp_height) else 0

        if self.__sqr_width == 0 or self.__sqr_height == 0:
            raise Exception("Width and height of dataframe are incorrect and cannot be used")

        self.__rows = list()
        self.__cols = list()
        self.__sqrs = list()

        self.__initialize_all_data(dataframe)

    def __initialize_all_data(self, dataframe) -> None:
        # TODO Check dims of dataframe

        for i in range(self.__height):
            self.__rows.append(Row(self.__height))
            self.__cols.append(Column(self.__width))
            self.__sqrs.append(Square(self.__sqr_width, self.__sqr_width))

        temp_squares = [[None]*self.__width for _ in range(self.__height)]
        for i in range(self.__height):
            for j in range(self.__width):
                
                sqr_dict = rc_to_sqr_idx(i, j, self.__width)
                sqr_idx, (sqr_row, sqr_col) = sqr_dict['sqr_index'], sqr_dict['sqr_local_rc']

                sqr_local_idx = rc_to_idx(sqr_row, sqr_col, self.__sqr_width)
                temp_squares[sqr_idx][sqr_local_idx] = Cell(dataframe.iat[i,j], index=rc_to_idx(i, j, self.__width), 
                                                            sqr_index=sqr_idx, sqr_row=sqr_row, sqr_col=sqr_col)

        # Now that the data is in the temp_squares variable in a manageable 
        # way, we need to insert it into our Squares objects
        for i,arr in enumerate(temp_squares):
            self.__sqrs[i].initialze_sqr(arr)


    def sparse_print(self, blanks='0') -> None:
        '''
        Print entire sudoku matrix with only values
        '''

        for row in self.__rows:
            for cell in row.get_cells():
                val = cell.get_value()
                print(f'{blanks if val == 0 else val}', end=' ')
            print()

    def __repr__(self) -> str:
        row_string = '\n\t'.join([repr(r) for r in self.__rows])
        col_string = '\n\t'.join([repr(c) for c in self.__cols])
        return f'rows: \n\t{row_string}\ncols: \n\t{col_string}\nsqrs: \n{self.__sqrs}' 



if __name__ == "__main__":
    # TODO Load unsolved sudoku in memory
    df = pd.read_csv("easy-1.csv")
    df = df.replace(np.nan, 0)
    df = df.astype(int)

    # TODO Setup sudoku structures
    setup_SQR_IDX_LOOKUP(9,9,3,3)
    sudoku = EntireSudoku(df)
    #print(sudoku)
    #sudoku.sparse_print(blanks='_')
    # TODO Solve sudoku 
    # TODO Print sudoku
