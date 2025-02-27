window.dash_clientside = Object.assign({}, window.dash_clientside, {
  clientside: {
    plot_update: function (data, layout) {
      return {
        data: data,
        layout: layout,
      };
    },
    progress_update: function (value) {
      return value;
    },
    theme_switched: (switchOn) => {
      document.documentElement.setAttribute(
        "data-bs-theme",
        switchOn ? "light" : "dark"
      );
      return window.dash_clientside.no_update;
    },
  },
});
