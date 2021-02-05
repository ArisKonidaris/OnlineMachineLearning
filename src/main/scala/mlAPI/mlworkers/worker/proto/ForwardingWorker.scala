package mlAPI.mlworkers.worker.proto

import BipartiteTopologyAPI.NodeInstance
import BipartiteTopologyAPI.annotations.{InitOp, MergeOp, ProcessOp, QueryOp}
import mlAPI.math.UsablePoint
import mlAPI.mlworkers.interfaces.Querier
import mlAPI.parameters.utils.ParameterDescriptor
import mlAPI.protocols.{CentralizedLearner, IntWrapper, LongWrapper, RemoteForwarder}

case class ForwardingWorker() extends NodeInstance[CentralizedLearner, Querier] with RemoteForwarder {

  override def poll(): Unit = ()

  @InitOp
  def init(): Unit = ()

  @ProcessOp
  def receiveTuple(data: UsablePoint): Unit = getProxy(0).forward(data)

  @QueryOp
  def query(qId: Long, qT: Int, pr: (Double, Array[UsablePoint])): Unit =
    getProxy(0)
      .describe(ParameterDescriptor(null, null, null, null, Array(LongWrapper(qId), IntWrapper(qT)), null))

  @MergeOp
  def merge(workers: Array[ForwardingWorker]): ForwardingWorker = this

}
