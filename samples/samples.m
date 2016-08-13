close all 
clear all

a=-1;
b=1;
j = 1;
       while i < 10000
           u1 = exp(-(a + (b - a) * rand));
           u2 = exp(-(a + (b - a) * rand));
           y1 = -log(u1);
           y2 = -log(u2);
           
           if y2>((y1-1)^2)/2
               z(j) = (y1) ;
               j = j + 1;
               i = i + 1;
           end
       end
       index = find(z>0.5);

%        index = find(z<=0.5);
%        z1=(z(index));
%        index1 = find(z>0.5);
%        z2=-abs(z(index1));
%        zz=[z1,z2];
       hist(z);