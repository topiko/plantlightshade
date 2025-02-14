include <../mirrors/surface.txt>;
$fa=1; $fs = .2;
t = 0.1;

module shade2d() {

    surf2 = [for (i=[len(surf)-1:-1:0]) [surf[i][0], surf[i][1] + t]];
    surf_ = concat(surf, surf2);
    echo(surf2);
    polygon(points=surf_);
}

module mirror3d() {
    rotate_extrude(angle=360) shade2d();
}

difference() {
    mirror3d();
    cylinder(h=200, d=4);
}


// shade2d();
