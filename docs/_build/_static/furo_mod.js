var backtotop = null;
var top_clicked = false;

const onScrollStop = (callback) => {
  let isScrolling;
  window.addEventListener(
    "scroll",
    (e) => {
      clearTimeout(isScrolling);
      isScrolling = setTimeout(() => {
        callback();
      }, 150);
    },
    false
  );
};

function adjustScrollPosition() {
  var scrollCurrentElement = document.querySelector(".scroll-current");
  var tocInstance = document.querySelector(".toc-tree-container");
  if (scrollCurrentElement && tocInstance) {
    var tocScrollTop = tocInstance.getBoundingClientRect().top; // Get the top offset of toc-scroll relative to the viewport
    var scrollCurrentTop = scrollCurrentElement.getBoundingClientRect().top; // Get the top offset of scroll-current relative to the viewport
    var tocScrollHeight = tocInstance.offsetHeight;
    var scrollToPosition =
      tocInstance.scrollTop +
      (scrollCurrentTop - tocScrollTop) -
      tocScrollHeight * 0.25; // Calculate the desired scroll position
    tocInstance.scrollTo(0, scrollToPosition); // Scroll toc-scroll to the desired position
  } else {
    console.log(
      "No scroll-current element found or tocInstance not initialized."
    );
    if (top_clicked) {
      top_clicked = false;
      if (tocInstance) {
        tocInstance.scrollTo(0, 0);
      }
    }
  }

  // if (tocLinkClicked) {
  //   tocLinkClicked = false;
  //   const tocLinks = document.querySelectorAll(".toc-tree a");
  //   tocLinks.forEach(function (link) {
  //     // Get target section id from href attribute
  //     // Remove active class from all tocLinks
  //     tocLinks.forEach(function (link) {
  //       link.closest("li").classList.remove(".scroll-current");
  //     });
  //     const url = link.getAttribute("href").substring(1);
  //     if (url == tocLinkUrl) {
  //       // Add active class to the clicked link
  //       link.closest("li").classList.add(".scroll-current");
  //       console.log("Current scroll changed to ", link.closest("li"));
  //     }
  //   });
  // }
}

document.addEventListener("DOMContentLoaded", () => {
  onScrollStop(adjustScrollPosition);
  backtotop = document.querySelector(".back-to-top");
  if (backtotop) {
    backtotop.addEventListener("click", () => {
      top_clicked = true;
      // var tocInstance = document.querySelector(".toc-tree-container");
      // if (tocInstance) {
      //   onScrollStop(() => {
      //     tocInstance.scrollTo(0, 0);
      //   });
      // }
    });
  }

  // Add click event listener to tocLinks
  // const tocLinks = document.querySelectorAll(".toc-tree a");
  // tocLinks.forEach(function (link) {
  //   link.addEventListener("click", function (event) {
  // event.preventDefault(); // Prevent default link behavior

  // Remove active class from all tocLinks
  // tocLinks.forEach(function (link) {
  //     link.closest("li").classList.remove(".scroll-current");
  // });

  // Add active class to the clicked link
  // link.closest("li").classList.add(".scroll-current");

  // Get target section id from href attribute
  // tocLinkUrl = link.getAttribute("href").substring(1);
  // tocLinkClicked = true;

  // Scroll to the target section
  // const targetSection = document.getElementById(targetId);
  // if (targetSection) {
  //     const headerHeight = document.querySelector("header").offsetHeight;
  //     const offset = targetSection.offsetTop - headerHeight;
  //     window.scrollTo({
  //         top: offset,
  //         behavior: "smooth"
  //     });
  // }
  //   });
  // });
});

//# sourceMappingURL=furo.js.map
