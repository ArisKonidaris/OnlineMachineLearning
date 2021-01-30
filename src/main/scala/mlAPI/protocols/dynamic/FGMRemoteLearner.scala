package mlAPI.protocols.dynamic

import BipartiteTopologyAPI.annotations.{RemoteOp, RemoteProxy}
import mlAPI.parameters.utils.ParameterDescriptor

@RemoteProxy
trait FGMRemoteLearner {

  @RemoteOp
  def newRound(model: ParameterDescriptor): Unit

  @RemoteOp
  def sendLocalDrift(): Unit

  @RemoteOp
  def requestZeta(): Unit

  @RemoteOp
  def receiveQuantum(quantum: Quantum): Unit

}
