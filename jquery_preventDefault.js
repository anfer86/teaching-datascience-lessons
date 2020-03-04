var scm = true;

$('body').bind('copy paste',function(e) {
    e.preventDefault(); return false;    
});

$('body').contextmenu(function() {
    return false;
});