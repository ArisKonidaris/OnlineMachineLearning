package mlAPI.parameters

import java.io

import mlAPI.learners.classification.trees.HoeffdingTree
import mlAPI.learners.classification.trees.serializable.HTDescriptor

case class HTParameters(ht: HoeffdingTree) extends LearningParameters {

  size = 0
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

  override def generateSerializedParams: (LearningParameters, Array[_]) => io.Serializable = {
    (lPar: LearningParameters, par: Array[_]) =>
      try {
        assert(lPar.isInstanceOf[HTParameters])
        if (par != null)
          if (par.length == 0)
            lPar.asInstanceOf[HTParameters].ht.serialize
          else
            lPar.asInstanceOf[HTParameters].ht.serialize
        else
          lPar.asInstanceOf[HTParameters].ht.serialize
      } catch {
        case _: Throwable =>
          throw new RuntimeException("Something happened while Serializing the HTParameters learning parameters.")
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

}
