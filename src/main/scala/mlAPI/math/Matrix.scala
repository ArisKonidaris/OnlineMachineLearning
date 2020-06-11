package mlAPI.math

/** Base trait for a matrix representation
  *
  */
trait Matrix {

  /** Number of rows
    *
    * @return number of rows request the matrix
    */
  def numRows: Int

  /** Number of columns
    *
    * @return number of columns request the matrix
    */
  def numCols: Int

  /** Element wise access function
    *
    * @param row row index
    * @param col column index
    * @return matrix entry at (row, col)
    */
  def apply(row: Int, col: Int): Double

  /** Element wise update function
    *
    * @param row   row index
    * @param col   column index
    * @param value value to set at (row, col)
    */
  def update(row: Int, col: Int, value: Double): Unit

  /** Copies the matrix instance
    *
    * @return Copy of itself
    */
  def copy: Matrix

  def equalsMatrix(matrix: Matrix): Boolean = {
    if (numRows == matrix.numRows && numCols == matrix.numCols) {
      val coordinates = for (row <- 0 until numRows; col <- 0 until numCols) yield (row, col)
      coordinates forall { case (row, col) => this.apply(row, col) == matrix(row, col) }
    } else {
      false
    }
  }

}
