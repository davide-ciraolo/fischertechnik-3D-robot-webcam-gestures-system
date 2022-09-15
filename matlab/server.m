[~,hostname] = system('hostname');
hostname = string(strtrim(hostname));
address = resolvehost(hostname,"address");
srv = tcpserver(address,7777,"ConnectionChangedFcn",@connectionFcn)
configureCallback(srv, "terminator", @connectionFcn)
global current_position;
current_position = [0; 0; 0; gripper.open];