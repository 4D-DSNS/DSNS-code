function O = DodecaElements

X(1,:) = [[-1/4-1/4*5^(1/2), 1/2, -1/4+1/4*5^(1/2)], [1/2, -1/4+1/4*5^(1/2),  1/4+1/4*5^(1/2)], [-1/4+1/4*5^(1/2), 1/4+1/4*5^(1/2), -1/2]];
X(2,:) = [[0, 1, 0], [0, 0, 1], [1, 0,  0]]; 
X(3,:) = [[1/4+1/4*5^(1/2), -1/2, -1/4+1/4*5^(1/2)], [-1/2, 1/4-1/4*5^(1/2), 1/4+1/4*5^(1/2)],  [1/4-1/4*5^(1/2), -1/4-1/4*5^(1/2), -1/2]];
X(4,:) = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]; 
X(5,:) = [[-1/4+1/4*5^ (1/2), -1/4-1/4*5^(1/2), -1/2], [1/4+1/4*5^(1/2), 1/2, 1/4-1/4*5^(1/2)], [1/2, 1/4-1/4*5^(1/2),  1/4+1/4*5^(1/2)]];
X(6,:) = [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]; 
X(7,:) = [[1/2, -1/4+1/4*5^(1/2), 1/4+1/4*5^ (1/2)], [-1/4+1/4*5^(1/2),1/4+1/4*5^(1/2), -1/2], [-1/4-1/4*5^(1/2), 1/2, -1/4+1/4*5^(1/2)]];
X(8,:) = [[-1/4+1/4*5^(1/2), -1/4-1/4*5^(1/2), 1/2], [1/4+1/4*5^(1/2), 1/2, -1/4+1/4*5^(1/2)], [-1/2,  -1/4+1/4*5^(1/2), 1/4+1/4*5^(1/2)]];
X(9,:) = [[0, 0, 1], [1, 0, 0], [0, 1, 0]];
X(10,:) = [[1/2, -1/4+1/4*5^ (1/2), -1/4-1/4*5^(1/2)], [1/4-1/4*5^(1/2), -1/4-1/4*5^(1/2), -1/2], [-1/4-1/4*5^(1/2), 1/2,  1/4-1/4*5^(1/2)]]; 
X(11,:) = [[-1/4-1/4*5^(1/2), -1/2, 1/4-1/4*5^(1/2)], [1/2, 1/4-1/4*5^(1/2), -1/4-1/4*5^ (1/2)], [-1/4+1/4*5^(1/2), -1/4-1/4*5^(1/2), 1/2]];
X(12,:) = [[1/4-1/4*5^(1/2), 1/4+1/4*5^(1/2),  -1/2], [-1/4-1/4*5^(1/2), -1/2, 1/4-1/4*5^(1/2)], [-1/2, -1/4+1/4*5^(1/2), 1/4+1/4*5^(1/2)]]; 
X(13,:) = [[1/4+1/4*5^(1/2), -1/2, -1/4+1/4*5^(1/2)], [1/2, -1/4+1/4*5^(1/2), -1/4-1/4*5^(1/2)], [-1/4+1/4*5^(1/2), 1/4+1/4*5^(1/2), 1/2]];
X(14,:) = [[1/4+1/4*5^(1/2), 1/2, 1/4-1/4*5^(1/2)], [1/2, 1/4-1/4*5^ (1/2), 1/4+1/4*5^(1/2)], [-1/4+1/4*5^(1/2), -1/4-1/4*5^(1/2), -1/2]];
X(15,:) = [[1/4+1/4*5^(1/2), -1/2,  1/4-1/4*5^(1/2)], [1/2,-1/4+1/4*5^(1/2), 1/4+1/4*5^(1/2)], [1/4-1/4*5^(1/2), -1/4-1/4*5^ (1/2), 1/2]]; 
X(16,:) = [[1/4+1/4*5^(1/2), -1/2, 1/4-1/4*5^(1/2)], [-1/2, 1/4-1/4*5^(1/2), -1/4-1/4*5^ (1/2)], [-1/4+1/4*5^(1/2), 1/4+1/4*5^(1/2), -1/2]];
X(17,:) = [[-1/2, -1/4+1/4*5^(1/2), -1/4-1/4*5^(1/2)],  [1/4-1/4*5^(1/2), 1/4+1/4*5^(1/2), 1/2], [1/4+1/4*5^(1/2), 1/2, 1/4-1/4*5^(1/2)]];
X(18,:) = [[1/4-1/4*5^ (1/2), -1/4-1/4*5^(1/2), -1/2], [1/4+1/4*5^(1/2), -1/2, -1/4+1/4*5^(1/2)], [-1/2, 1/4-1/4*5^ (1/2), 1/4+1/4*5^(1/2)]];
X(19,:) = [[-1/4-1/4*5^(1/2), 1/2, 1/4-1/4*5^(1/2)], [1/2, -1/4+1/4*5^(1/2),  -1/4-1/4*5^(1/2)], [1/4-1/4*5^(1/2), -1/4-1/4*5^(1/2), -1/2]];
X(20,:) = [[-1/4+1/4*5^(1/2), 1/4+1/4*5^ (1/2), -1/2], [-1/4-1/4*5^(1/2), 1/2, -1/4+1/4*5^(1/2)], [1/2, -1/4+1/4*5^(1/2), 1/4+1/4*5^ (1/2)]]; 
X(21,:) = [[-1/4+1/4*5^(1/2), 1/4+1/4*5^(1/2), -1/2], [1/4+1/4*5^(1/2), -1/2, 1/4-1/4*5^(1/2)],  [-1/2, 1/4-1/4*5^(1/2),-1/4-1/4*5^(1/2)]];
X(22,:) = [[1/4+1/4*5^(1/2), 1/2, -1/4+1/4*5^(1/2)], [-1/2,  -1/4+1/4*5^(1/2), 1/4+1/4*5^(1/2)], [-1/4+1/4*5^(1/2), -1/4-1/4*5^(1/2), 1/2]]; 
X(23,:) = [[1/2, -1/4+1/4*5^(1/2), -1/4-1/4*5^(1/2)], [-1/4+1/4*5^(1/2), 1/4+1/4*5^(1/2), 1/2], [1/4+1/4*5^(1/2),  -1/2, -1/4+1/4*5^(1/2)]];
X(24,:) = [[-1/2, 1/4-1/4*5^(1/2), -1/4-1/4*5^(1/2)], [-1/4+1/4*5^(1/2), 1/4+1/4*5^(1/2), -1/2], [1/4+1/4*5^(1/2), -1/2, 1/4-1/4*5^(1/2)]];
X(25,:) = [[-1/2, -1/4+1/4*5^(1/2),  -1/4-1/4*5^(1/2)], [-1/4+1/4*5^(1/2), -1/4-1/4*5^(1/2), -1/2], [-1/4-1/4*5^(1/2), -1/2, -1/4+1/4*5^ (1/2)]];
X(26,:) = [[1/2, 1/4-1/4*5^(1/2), 1/4+1/4*5^(1/2)], [1/4-1/4*5^(1/2), 1/4+1/4*5^(1/2), 1/2],  [-1/4-1/4*5^(1/2), -1/2,-1/4+1/4*5^(1/2)]];
X(27,:) = [[-1/2, -1/4+1/4*5^(1/2), 1/4+1/4*5^(1/2)],  [1/4-1/4*5^(1/2), 1/4+1/4*5^(1/2), -1/2], [-1/4-1/4*5^(1/2), -1/2, 1/4-1/4*5^(1/2)]]; 
X(28,:) = [[-1/4+1/4*5^ (1/2), -1/4-1/4*5^(1/2), 1/2], [-1/4-1/4*5^(1/2), -1/2, 1/4-1/4*5^(1/2)], [1/2, 1/4-1/4*5^ (1/2), -1/4-1/4*5^(1/2)]];
X(29,:) = [[0, -1, 0], [0, 0, -1], [1, 0, 0]];
X(30,:) = [[-1/4+1/4*5^(1/2), 1/4+1/4*5^ (1/2), 1/2], [-1/4-1/4*5^(1/2), 1/2, 1/4-1/4*5^(1/2)], [-1/2, 1/4-1/4*5^(1/2), 1/4+1/4*5^ (1/2)]]; 
X(31,:) = [[1/4-1/4*5^(1/2), -1/4-1/4*5^(1/2), 1/2], [1/4+1/4*5^(1/2), -1/2, 1/4-1/4*5^(1/2)], [1/2,  -1/4+1/4*5^(1/2), 1/4+1/4*5^(1/2)]];
X(32,:) = [[1/2, 1/4-1/4*5^(1/2), -1/4-1/4*5^(1/2)],[1/4-1/4*5^ (1/2), 1/4+1/4*5^(1/2), -1/2], [1/4+1/4*5^(1/2), 1/2, -1/4+1/4*5^(1/2)]];
X(33,:) = [[-1/4-1/4*5^(1/2),  1/2, -1/4+1/4*5^(1/2)],[-1/2, 1/4-1/4*5^(1/2), -1/4-1/4*5^(1/2)], [1/4-1/4*5^(1/2),  -1/4-1/4*5^(1/2), 1/2]];
X(34,:) = [[-1/2,1/4-1/4*5^(1/2), 1/4+1/4*5^(1/2)], [1/4-1/4*5^(1/2), -1/4-1/4*5^ (1/2), -1/2], [1/4+1/4*5^(1/2), -1/2, -1/4+1/4*5^(1/2)]];
X(35,:) = [[1/4-1/4*5^(1/2), -1/4-1/4*5^(1/2),  -1/2], [-1/4-1/4*5^(1/2), 1/2, 1/4-1/4*5^(1/2)], [1/2, -1/4+1/4*5^(1/2), -1/4-1/4*5^(1/2)]];  
X(36,:) = [[1/4+1/4*5^(1/2), 1/2, 1/4-1/4*5^(1/2)], [-1/2, -1/4+1/4*5^(1/2), -1/4-1/4*5^(1/2)], [1/4-1/4*5^ (1/2), 1/4+1/4*5^(1/2),1/2]];
X(37,:) = [[-1/4-1/4*5^(1/2), -1/2, -1/4+1/4*5^(1/2)], [-1/2, -1/4+1/4*5^ (1/2), -1/4-1/4*5^(1/2)], [-1/4+1/4*5^(1/2), -1/4-1/4*5^(1/2), -1/2]]; 
X(38,:) = [[1/4-1/4*5^(1/2), 1/4+1/4*5^(1/2), 1/2], [1/4+1/4*5^(1/2), 1/2, 1/4-1/4*5^(1/2)], [-1/2, -1/4+1/4*5^(1/2),  -1/4-1/4*5^(1/2)]];
X(39,:) = [[1/4-1/4*5^(1/2), -1/4-1/4*5^(1/2), 1/2], [-1/4-1/4*5^(1/2), 1/2, -1/4+1/4*5^ (1/2)], [-1/2, 1/4-1/4*5^(1/2), -1/4-1/4*5^(1/2)]]; 
X(40,:) = [[-1/2, 1/4-1/4*5^(1/2), 1/4+1/4*5^(1/2)],  [-1/4+1/4*5^(1/2), 1/4+1/4*5^(1/2), 1/2], [-1/4-1/4*5^(1/2), 1/2, 1/4-1/4*5^(1/2)]];  
X(41,:) = [[-1/4-1/4*5^(1/2), -1/2, -1/4+1/4*5^(1/2)], [1/2, 1/4-1/4*5^(1/2), 1/4+1/4*5^(1/2)], [1/4-1/4*5^ (1/2), 1/4+1/4*5^(1/2), 1/2]];
X(42,:) = [[-1/2, -1/4+1/4*5^(1/2), 1/4+1/4*5^(1/2)], [-1/4+1/4*5^(1/2),  -1/4-1/4*5^(1/2), 1/2], [1/4+1/4*5^(1/2), 1/2, -1/4+1/4*5^(1/2)]]; 
X(43,:) = [[-1/4+1/4*5^(1/2), 1/4+1/4*5^(1/2), 1/2], [1/4+1/4*5^(1/2), -1/2, -1/4+1/4*5^(1/2)], [1/2, -1/4+1/4*5^(1/2), -1/4-1/4*5^ (1/2)]];
X(44,:) = [[1/2, 1/4-1/4*5^(1/2), 1/4+1/4*5^(1/2)], [-1/4+1/4*5^(1/2), -1/4-1/4*5^(1/2),  -1/2], [1/4+1/4*5^(1/2), 1/2,1/4-1/4*5^(1/2)]]; 
X(45,:) = [[0, -1, 0], [0, 0, 1], [-1, 0, 0]];
X(46,:) = [[1/4-1/4*5^ (1/2), 1/4+1/4*5^(1/2), -1/2], [1/4+1/4*5^(1/2), 1/2, -1/4+1/4*5^(1/2)],[1/2, 1/4-1/4*5^ (1/2), -1/4-1/4*5^(1/2)]]; 
X(47,:) = [[1, 0, 0], [0, -1, 0], [0, 0, -1]];
X(48,:) = [[0, 0, 1], [-1, 0, 0], [0, -1,  0]]; 
X(49,:) = [[1/2, -1/4+1/4*5^(1/2), 1/4+1/4*5^(1/2)], [1/4-1/4*5^(1/2), -1/4-1/4*5^(1/2), 1/2], [1/4+1/4*5^(1/2), -1/2, 1/4-1/4*5^(1/2)]]; 
X(50,:) = [[-1/4-1/4*5^(1/2), 1/2, 1/4-1/4*5^(1/2)], [-1/2,  1/4-1/4*5^(1/2), 1/4+1/4*5^(1/2)], [-1/4+1/4*5^(1/2), 1/4+1/4*5^(1/2), 1/2]]; 
X(51,:) = [[1/4-1/4*5^(1/2),  1/4+1/4*5^(1/2), 1/2], [-1/4-1/4*5^(1/2), -1/2, -1/4+1/4*5^(1/2)], [1/2, 1/4-1/4*5^(1/2), 1/4+1/4*5^(1/2)]]; 
X(52,:) = [[0, 0, -1], [1, 0, 0], [0, -1, 0]];
X(53,:) = [[-1/4-1/4*5^(1/2), -1/2, 1/4-1/4*5^ (1/2)], [-1/2, -1/4+1/4*5^(1/2), 1/4+1/4*5^(1/2)], [1/4-1/4*5^(1/2), 1/4+1/4*5^(1/2), -1/2]];  
X(54,:) = [[-1, 0, 0], [0, 1, 0], [0,0, -1]]; 
X(55,:) = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]];
X(56,:) = [[-1/2, 1/4-1/4*5^ (1/2), -1/4-1/4*5^(1/2)], [1/4-1/4*5^(1/2), -1/4-1/4*5^(1/2), 1/2], [-1/4-1/4*5^(1/2), 1/2, -1/4+1/4*5^(1/2)]]; 
X(57,:) = [[-1/4+1/4*5^(1/2), -1/4-1/4*5^(1/2), -1/2], [-1/4-1/4*5^(1/2), -1/2, -1/4+1/4*5^(1/2)], [-1/2, -1/4+1/4*5^(1/2), -1/4-1/4*5^(1/2)]]; 
X(58,:) = [[0, 1, 0], [0, 0, -1], [-1, 0, 0]];  
X(59,:) = [[1/4+1/4*5^(1/2), 1/2, -1/4+1/4*5^(1/2)], [1/2, 1/4-1/4*5^(1/2), -1/4-1/4*5^(1/2)],  [1/4-1/4*5^(1/2), 1/4+1/4*5^(1/2), -1/2]]; 
X(60,:) = [[1/2, 1/4-1/4*5^(1/2), -1/4-1/4*5^(1/2)], [-1/4+1/4*5^ (1/2), -1/4-1/4*5^(1/2), 1/2], [-1/4-1/4*5^(1/2), -1/2, 1/4-1/4*5^(1/2)]];

for j=1:60
    O{j}=[X(j,1:3);X(j,4:6);X(j,7:9)];
end