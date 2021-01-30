package mlAPI.parameters

import mlAPI.learners.classification.trees.HoeffdingTree
import mlAPI.learners.classification.trees.serializable.HTDescriptor
import mlAPI.parameters.utils.{ParameterDescriptor, SerializableParameters, WrappedStructuredParameters}

case class HTParameters(ht: HoeffdingTree) extends LearningParameters {

  bytes = {
    ht.calculateSize()
    ht.getSize
  }

  def this() = this(null)

  override def getCopy: LearningParameters = {
    HTParameters({
      val newHT: HoeffdingTree = new HoeffdingTree()
      newHT.deserialize(ht.serialize)
      newHT
    })
  }

  override def extractParams: (LearningParameters, Boolean) => SerializableParameters = {
    (params: LearningParameters, _: Boolean) =>
      WrappedStructuredParameters(params.asInstanceOf[HTParameters].ht.serialize)
  }

  override def extractDivParams: (LearningParameters, Array[_]) => Array[Array[SerializableParameters]] = {
    (params: LearningParameters, args: Array[_]) =>
      try {
        assert(params.isInstanceOf[HTParameters])
        if (args != null)
          if (args.length == 0)
            Array(Array(utils.WrappedStructuredParameters(params.asInstanceOf[HTParameters].ht.serialize)))
          else
            Array(Array(utils.WrappedStructuredParameters(params.asInstanceOf[HTParameters].ht.serialize)))
        else
          Array(Array(utils.WrappedStructuredParameters(params.asInstanceOf[HTParameters].ht.serialize)))
      } catch {
        case _: Throwable =>
          throw new RuntimeException("Something happened while extracting the HTParameters learning parameters.")
      }
  }

  override def generateParameters(pDesc: ParameterDescriptor): LearningParameters = {
    try {
      assert(pDesc.dataStructure.isInstanceOf[HTDescriptor])
      val tree: HoeffdingTree = new HoeffdingTree()
      tree.deserialize(pDesc.dataStructure.asInstanceOf[HTDescriptor])
      HTParameters(tree)
    } catch {
      case _: Throwable =>
        throw new RuntimeException("Something happened while deserializing the HTParameters learning parameters.")
    }
  }

  override def equals(obj: Any): Boolean = {
    obj match {
      case htp: HTParameters => ht.serialize.toString.equals(htp.ht.serialize.toString)
      case _ => false
    }
  }

}
