package mlAPI.dataBuffers

import scala.collection.mutable.ListBuffer

object DatasetUtils {
  /**
   * Tail recursive method for merging two data buffers.
   */
  @scala.annotation.tailrec
  def mergeBufferedPoints[T](count1: Int, size1: Int,
                             count2: Int, size2: Int,
                             set1: ListBuffer[T], set2: ListBuffer[T],
                             offset: Int): ListBuffer[T] = {
    if (count2 == size2) {
      set1
    } else if (count1 == size1) {
      set1 ++ set2
    } else {
      set1.insert(count1, set2(count2))
      mergeBufferedPoints(count1 + 1 + offset, size1, count2 + 1, size2, set1, set2, offset)
    }
  }
}
