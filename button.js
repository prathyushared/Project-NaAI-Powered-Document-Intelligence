// src/components/ui/Button.js
import React from 'react';

export default function Button({ children, className, variant = 'solid', ...props }) {
  // Define the base styles for the button
  const baseStyles = 'px-4 py-2 rounded-md font-semibold focus:outline-none transition duration-200';

  // Define the styles for different variants
  const variantStyles = variant === 'outline'
    ? 'bg-transparent border-2 border-indigo-600 text-indigo-600 hover:bg-indigo-100'
    : 'bg-indigo-600 text-white hover:bg-indigo-700';

  return (
    <button className={`${baseStyles} ${variantStyles} ${className}`} {...props}>
      {children}
    </button>
  );
}
