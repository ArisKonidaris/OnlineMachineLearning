package mlAPI.protocols.dynamic

/** A Serializable trait of a value. */
trait ValueContainer[T] extends java.io.Serializable {

  def getValue: T

  def setValue(value: T): Unit

}
