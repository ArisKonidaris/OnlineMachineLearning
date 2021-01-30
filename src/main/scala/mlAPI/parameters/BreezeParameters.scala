package mlAPI.parameters

import breeze.linalg.{DenseVector => BreezeDenseVector}
import mlAPI.math.{DenseVector, SparseVector, Vector}
import mlAPI.parameters.utils.Bucket

import scala.collection.mutable.ListBuffer

/**
  * A trait for determining if the Learning Parameters are
  * represented by the Breeze library.
  */
trait BreezeParameters extends VectoredParameters {

  def flatten: BreezeDenseVector[Double]

  override def toDenseVector: Vector = DenseVector.denseVectorConverter.convert(flatten)

  override def toSparseVector: Vector = SparseVector.sparseVectorConverter.convert(flatten)

  def unwrapData(sizes: Array[Int], data: Array[Double]): Array[Array[Double]] = {
    require(sizes.sum == data.length, "Not valid bucket and data given to unwrapData function.")

    @scala.annotation.tailrec
    def recursiveUnwrapping(sz: Array[Int], dt: Array[Double], result: ListBuffer[Array[Double]])
    : ListBuffer[Array[Double]] = {
      if (sz.isEmpty) {
        result
      } else {
        result.append(dt.slice(0, sz.head))
        recursiveUnwrapping(sz.tail, dt.slice(sz.head, dt.length), result)
      }
    }

    recursiveUnwrapping(sizes, data, new ListBuffer[Array[Double]]).toArray
  }

  override def slice(range: Bucket, sparse: Boolean = false): Vector = {
    sliceRequirements(range)
//    val flatVector: BreezeDenseVector[Double] = {
//      if (range.getLength == size)
//        flatten
//      else
//        flatten(range.getStart.toInt to range.getEnd.toInt).copy
//    }
//    if (sparse)
//      SparseVector.sparseVectorConverter.convert(flatVector)
//    else
//      DenseVector.denseVectorConverter.convert(flatVector)
    val flatVector: Array[Double] = {
      if (range.getLength == size)
        flatten.data
      else
        flatten.data.slice(range.getStart.toInt, range.getEnd.toInt + 1)
    }
    if (sparse)
      DenseVector(flatVector).toSparseVector
    else
      DenseVector(flatVector)
  }

  override def frobeniusNorm: Double = breeze.linalg.norm(flatten)

}
