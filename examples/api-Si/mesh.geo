Point(0) = {5.0,5.0,-1.5,1.0};
Point(1) = {-5.0,5.0,-1.5,1.0};
Point(2) = {-5.0,-5.0,-1.5,1.0};
Point(3) = {5.0,-5.0,-1.5,1.0};
Line(1) = {0,1};
Line(2) = {1,2};
Line(3) = {2,3};
Line(4) = {3,0};
Line Loop(0) = {1,2,3,4};
Point(4) = {5.0,5.0,1.5,1.0};
Point(5) = {-5.0,5.0,1.5,1.0};
Point(6) = {-5.0,-5.0,1.5,1.0};
Point(7) = {5.0,-5.0,1.5,1.0};
Line(5) = {4,5};
Line(6) = {5,6};
Line(7) = {6,7};
Line(8) = {7,4};
Line Loop(1) = {5,6,7,8};
Line(9) = {0,4};
Line(10) = {1,5};
Line(11) = {2,6};
Line(12) = {3,7};
Line Loop(2) = {1,10,-5,-9};
Plane Surface(8) = {2};
Line Loop(3) = {2,11,-6,-10};
Plane Surface(9) = {3};
Line Loop(4) = {3,12,-7,-11};
Plane Surface(10) = {4};
Line Loop(5) = {4,9,-8,-12};
Plane Surface(11) = {5};
Point(8) = {-0.5,-0.5,-1.5,1.0};
Point(9) = {-0.5,0.5,-1.5,1.0};
Point(10) = {0.5,0.5,-1.5,1.0};
Point(11) = {0.5,-0.5,-1.5,1.0};
Line(13) = {8,9};
Line(14) = {9,10};
Line(15) = {10,11};
Line(16) = {11,8};
Line Loop(6) = {13,14,15,16};
Point(12) = {-0.5,-0.5,1.5,1.0};
Point(13) = {-0.5,0.5,1.5,1.0};
Point(14) = {0.5,0.5,1.5,1.0};
Point(15) = {0.5,-0.5,1.5,1.0};
Line(17) = {12,13};
Line(18) = {13,14};
Line(19) = {14,15};
Line(20) = {15,12};
Line Loop(7) = {17,18,19,20};
Line(21) = {8,12};
Line(22) = {9,13};
Line(23) = {10,14};
Line(24) = {11,15};
Line Loop(8) = {13,22,-17,-21};
Plane Surface(20) = {8};
Line Loop(9) = {14,23,-18,-22};
Plane Surface(21) = {9};
Line Loop(10) = {15,24,-19,-23};
Plane Surface(22) = {10};
Line Loop(11) = {16,21,-20,-24};
Plane Surface(23) = {11};
Plane Surface(24) = {1,7};
Plane Surface(25) = {0,6};
Periodic Surface 11 {4,9,-8,-12} = 9 {-2,10,6,-11};
Physical Surface("Periodic_1") = {11};
Physical Surface("Periodic_2") = {9};
Periodic Surface 8 {1,10,-5,-9} = 10 {-3,11,7,-12};
Physical Surface("Periodic_3") = {8};
Physical Surface("Periodic_4") = {10};
Physical Surface('Boundary') = {20,21,22,23,25,24};
Surface Loop(26) = {8,9,10,11,20,21,22,23,25,24};
Volume(0) = {26};
Physical Volume('Bulk') = {0};