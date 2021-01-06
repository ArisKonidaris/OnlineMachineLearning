package mlAPI.parameters

import ControlAPI.CountableSerial

/** Bucket for distributing the parameters to multiple parameter servers.
 *
 * @param start The first parameter.
 * @param end   The last parameter.
 */
case class Bucket(protected var start: Long, protected var end: Long) extends CountableSerial {

  require(start >= 0 && start <= end)

  def this() = this(0, 0)

  def getRangeLength: Long = end - start + 1

  def getStart: Long = start

  def getEnd: Long = end

  def setStart(start: Long): Unit = this.start = start

  def setEnd(end: Long): Unit = this.end = end

  override def getSize: Int = 16

}
