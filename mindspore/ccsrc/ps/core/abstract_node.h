/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_CCSRC_PS_CORE_ABSTRACT_NODE_H_
#define MINDSPORE_CCSRC_PS_CORE_ABSTRACT_NODE_H_

#include <utility>
#include <string>
#include <memory>
#include <map>
#include <vector>
#include <unordered_map>

#include "ps/core/node.h"

namespace mindspore {
namespace ps {
namespace core {
class AbstractNode : public Node {
 public:
  AbstractNode() : heart_beat_thread_(nullptr), client_to_scheduler_thread_(nullptr), client_to_scheduler_(nullptr) {}
  ~AbstractNode() override = default;

  bool BroadcastToServers(const std::string &message, const uint32_t &timeout = kCommTimeoutInSeconds);
  void set_event_callback(const OnNodeEventMessage &on_node_event_message);

  virtual bool Send(const enum NodeRole &node_role, const uint32_t &rank_id, const std::string &message,
                    const uint32_t &timeout = kCommTimeoutInSeconds);
  virtual bool Send(const NodeRole &node_role, const std::vector<uint32_t> &rank_ids,
                    const std::vector<std::string> &data, const uint32_t &timeout = kCommTimeoutInSeconds);
  virtual bool Send(const enum NodeRole &node_role, const uint32_t &rank_id, const std::string &message,
                    CommMessage *comm_message_resp, const uint32_t &timeout = kCommTimeoutInSeconds);
  virtual bool Send(const NodeRole &node_role, const std::vector<uint32_t> &rank_ids,
                    const std::vector<std::string> &data, std::vector<CommMessage> *comm_message_resp,
                    const uint32_t &timeout = kCommTimeoutInSeconds);

  bool Wait(uint64_t request_id, const uint32_t &timeout = kCommTimeoutInSeconds);

 protected:
  void Register(const std::shared_ptr<TcpClient> &client);
  void ProcessRegisterResp(const CommMessage &message);
  void StartHeartbeatTimer(const std::shared_ptr<TcpClient> &client);
  void Heartbeat(const std::shared_ptr<TcpClient> &client, bool is_node_finish = false);
  void ProcessHeartbeatResp(const CommMessage &message);
  void FetchServers(const std::shared_ptr<TcpClient> &client);
  void ProcessFetchServersResp(const CommMessage &message);
  bool Disconnect(const std::shared_ptr<TcpClient> &client, const uint32_t &timeout);
  bool WaitForDisconnect(const uint32_t &timeout);
  bool InitClientToScheduler();
  const std::shared_ptr<TcpClient> &GetOrCreateTcpClient(const int &rank_id);
  bool SendMessageSync(const std::shared_ptr<TcpClient> &client, const CommMessage &message,
                       const uint32_t &timeout = kCommTimeoutInSeconds);
  void SendMessageAsync(const std::shared_ptr<TcpClient> &client, const CommMessage &message);
  void ProcessSendDataResp(const CommMessage &message);
  void RunMessageCallback(const uint64_t &request_id);
  void set_message_callback(const uint64_t &request_id, const MessageCallback &message_callback);
  void NotifyMessageArrival(const CommMessage &message);

  std::unique_ptr<std::thread> heart_beat_thread_;
  std::unique_ptr<std::thread> client_to_scheduler_thread_;
  std::shared_ptr<TcpClient> client_to_scheduler_;

  OnNodeEventMessage on_node_event_message_;
  // the map's key is: <node_role,rank_id>, the map's value is: <ip, port>
  std::map<std::pair<NodeRole, uint32_t>, std::pair<std::string, uint16_t>> nodes_address_;
  std::mutex client_mutex_;
  // the map's key is: rank_id
  std::unordered_map<int, std::shared_ptr<TcpClient>> connected_nodes_;

  // the map's key is: request_id, the map's value is: <expected responses, actual responses>
  std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> message_tracker_;
  std::mutex message_tracker_mutex_;
  std::condition_variable message_tracker_cond_;

  // the map's key is: request_id, the map's value is:<rank_id, CommMessage>
  std::unordered_map<uint64_t, std::unordered_map<uint32_t, CommMessage>> receive_messages_;
  std::mutex receive_messages_mutex_;
  // the map's key is: request_id
  std::unordered_map<uint64_t, MessageCallback> message_callbacks_;
  std::mutex message_callbacks_mutex_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_ABSTRACT_NODE_H_
